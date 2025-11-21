# Process Curry Video for Benchmark

This guide explains how to use the Curry shot video from your Downloads folder to create a benchmark with pose overlay.

## Steps

1. **Open the processing page**
   - Open `process_curry_video.html` in your browser
   - This page provides a simple interface to process the Curry video

2. **Select the Curry video**
   - Click the upload area or drag and drop "refined_curry_benchmark_with_overlay.MOV" from your Downloads folder
   - The video will be previewed

3. **Process the video**
   - Click "Process Video" button
   - Wait for processing to complete (this may take a minute)
   - The video will be processed frame by frame with MediaPipe Pose detection
   - Pose overlays will be applied to each frame

4. **Download the benchmark file**
   - Once processing is complete, `curry_benchmark.js` will automatically download
   - Move this file to the `player_data/` folder (replacing the placeholder)

5. **Use the benchmark**
   - The app will now use the real Curry benchmark data instead of synthetic data
   - When users select Stephen Curry, they'll compare against the actual shot from the video

## How it works

The `processVideoForBenchmark()` function:
- Loads the video file
- Processes each frame with MediaPipe Pose
- Detects shooting motion stages (neutral → pre_shot → follow_through)
- Extracts pose landmarks, angles (elbow, wrist, arm), and applies overlay
- Generates benchmark data in the same format as `lebron_benchmark.js`
- Downloads the benchmark file automatically

## Alternative: Console Method

You can also process the video directly from the browser console:

```javascript
// Create a file input
const input = document.createElement('input');
input.type = 'file';
input.accept = 'video/*';
input.onchange = async (e) => {
    const file = e.target.files[0];
    await processVideoForBenchmark(file, 'curry');
};
input.click();
```

## Notes

- The video must show a clear shooting motion with full body visible
- Processing time depends on video length (typically 30-60 seconds)
- The generated benchmark data includes pose landmarks with overlay information
- If no shot is detected, try a different video or ensure the person is fully visible

