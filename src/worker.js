import { pipeline, env } from '@huggingface/transformers';

/* 
  💡 SWITCH TO OFFLINE LOCAL MODE INSTEAD OF HUGGINGFACE CDN 
  https://huggingface.co/docs/transformers.js/en/custom_usage
*/
// Specify a custom location for models (defaults to '/models/').
env.localModelPath = '/models/';
// Disable the loading of remote models from the Hugging Face Hub:
env.allowRemoteModels = false;
// Set location of .wasm files. Defaults to use a CDN.
env.backends.onnx.wasm.wasmPaths = '/models/';
env.allowLocalModels = true;

// Skip local model check (REMOTE)
// env.allowLocalModels = false;

// Use the Singleton pattern to enable lazy construction of the pipeline.
class PipelineSingleton {
  static task = 'feature-extraction';
  static model = 'Supabase/gte-small';
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, {
        progress_callback,
        dtype: 'fp32',
        device: navigator.gpu ? 'webgpu' : 'wasm',
        // ESLint: Redundant double negation.eslintno-extra-boolean-cast
        // device: !!navigator.gpu ? 'webgpu' : 'wasm',
      });
    }
    return this.instance;
  }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
  // Retrieve the classification pipeline. When called for the first time,
  // this will load the pipeline and save it for future use.
  let classifier = await PipelineSingleton.getInstance((x) => {
    // We also add a progress callback to the pipeline so that we can
    // track model loading.
    self.postMessage(x);
  });

  // Actually perform the classification
  let output = await classifier(event.data.text, {
    pooling: 'mean',
    normalize: true,
  });

  // Extract the embedding output
  const embedding = Array.from(output.data);

  // Send the output back to the main thread
  self.postMessage({
    status: 'complete',
    embedding,
  });
});
