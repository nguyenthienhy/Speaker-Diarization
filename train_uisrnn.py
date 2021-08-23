from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from functools import partial
import uisrnn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
mp = mp.get_context('forkserver')

NUM_WORKERS = 4

def diarization_experiment(model_args, training_args, inference_args):
    """Experiment pipeline.

    Load dataset --> train model --> test model --> output result

    Args:
      model_args: model configurations
      training_args: training configurations
      inference_args: inference configurations
    """

    # Train data
    train_sequences = np.load('./data/data_embedding/training_data_0821.npz', 
                            allow_pickle=True)['train_sequence'][0:3000]
    train_sequences = [seq.astype(float) + 0.00001 for seq in train_sequences]                        
    train_cluster_ids = np.load('./data/data_embedding/training_data_0821.npz', 
                            allow_pickle=True)['train_cluster_id'][0:3000]
    train_cluster_ids = [np.array(cid).astype(str) for cid in train_cluster_ids]

    # Training
    model = uisrnn.UISRNN(model_args)
    writer = SummaryWriter()
    for epoch in range(training_args.epochs):
      print("Epochs: " + str(epoch))
      stats = model.fit(train_sequences, train_cluster_ids, training_args)
      # add to tensorboard
      for loss, cur_iter in stats:
        for loss_name, loss_value in loss.items():
          writer.add_scalar('loss/' + loss_name, loss_value, cur_iter)
    model.save('saved_model_uisrnn/model_en_0821.uisrnn')
    model.load('saved_model_uisrnn/model_en_0821.uisrnn') 

    # Test data
    test_sequences = np.load('./data/data_embedding/training_data_0821.npz', 
                            allow_pickle=True)['train_sequence'][6000:]
    test_sequences = [seq.astype(float) + 0.00001 for seq in test_sequences]                        
    test_cluster_ids = np.load('./data/data_embedding/training_data_0821.npz', 
                            allow_pickle=True)['train_cluster_id'][6000:]
    test_cluster_ids = [np.array(cid).astype(str) for cid in test_cluster_ids]

    # Testing.
    predicted_cluster_ids = []
    test_record = []
    # predict sequences in parallel
    model.rnn_model.share_memory()
    pool = mp.Pool(NUM_WORKERS, maxtasksperchild=None)
    pred_gen = pool.imap(
        func=partial(model.predict, args=inference_args),
        iterable=test_sequences)
    # collect and score predicitons
    for idx, predicted_cluster_id in enumerate(pred_gen):
      accuracy = uisrnn.compute_sequence_match_accuracy(
          test_cluster_ids[idx], predicted_cluster_id)
      predicted_cluster_ids.append(predicted_cluster_id)
      test_record.append((accuracy, len(test_cluster_ids[idx])))
      print('Ground truth labels:')
      print(test_cluster_ids[idx])
      print('Predicted labels:')
      print(predicted_cluster_id)
      print('-' * 80)
    # close multiprocessing pool
    pool.close()
    # close tensorboard writer
    writer.close()
    print(uisrnn.output_result(model_args, training_args, test_record))


def train():

    """The train function."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    model_args.rnn_depth = 1
    model_args.rnn_hidden_size = 512

    training_args.enforce_cluster_id_uniqueness = False
    training_args.batch_size = 16
    training_args.learning_rate = 5e-5
    training_args.train_iteration = 110
    training_args.num_permutations = 0 # if set higher, maybe need higher ram
    training_args.loss_samples = 1
    training_args.learning_rate_half_life = 0
    training_args.epochs = 2

    diarization_experiment(model_args, training_args, inference_args)

# if __name__ == '__main__':
#     train()
