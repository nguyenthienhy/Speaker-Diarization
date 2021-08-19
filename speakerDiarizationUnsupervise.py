from embedding import cluster_utils, consts, new_utils, toolkits
from visualization.viewer import PlotDiar

toolkits.initialize_GPU(consts.nn_params.gpu)

wav = "sample/ahnss.wav"
specs, intervals = new_utils.slide_window(audio_path=wav,
                                          embedding_per_second=consts.slide_window_params.embedding_per_second,
                                          overlap_rate=consts.slide_window_params.overlap_rate)

embeddings = new_utils.generate_embeddings(specs)
embeddings = cluster_utils.umap_transform(embeddings)

predicted_labels = cluster_utils.cluster_by_hdbscan(embeddings)

reference = new_utils.reference_rttm("sample/ahnss.rttm")
hypothesis = new_utils.result_map(intervals, predicted_labels)

der = new_utils.der(reference, hypothesis)

new_utils.save_and_export(result_map=hypothesis, 
                          dir=consts.audio_dir, 
                          audio_name="ahnss.wav", 
                          der=der['diarization error rate'])
