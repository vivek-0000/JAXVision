// Shapes of the JSON files written by the Python side (metrics.json, benchmark.json).

export interface EpochRow {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  test_loss: number;
  test_accuracy: number;
  learning_rate: number;
  epoch_time_s: number;
  images_per_sec: number;
}

export interface Metrics {
  config: Record<string, unknown>;
  dataset: { name: string; source: string; train_size: number; test_size: number; class_names: string[] };
  hardware: { backend: string; device_kind: string; num_devices: number; global_batch_size: number };
  model: { parameters: number; widths: number[] };
  history: EpochRow[];
  summary: {
    epochs: number;
    final_test_accuracy: number;
    best_test_accuracy: number;
    final_train_accuracy: number;
    total_train_time_s: number;
    mean_images_per_sec: number;
  };
}

export interface Benchmark {
  backend: string;
  device_kind: string;
  num_devices: number;
  inference_jit_vs_eager: {
    batch_size: number;
    compile_time_s: number;
    eager_ms: number;
    jit_ms: number;
    speedup: number;
    jit_images_per_sec: number;
  };
  inference_batch_sweep: { batch_size: number; latency_ms: number; images_per_sec: number }[];
  training_scaling: {
    devices: number;
    global_batch: number;
    step_ms: number;
    images_per_sec: number;
    speedup_vs_1_device: number;
    efficiency: number;
  }[];
}

export interface ModelInfo {
  dataset: string;
  classes: string[];
  image_shape: number[];
  parameters: number;
  widths: number[];
}

export interface Sample {
  image: string; // data: URL
  label: string;
}

export interface Prediction {
  label: string;
  probability: number;
}
