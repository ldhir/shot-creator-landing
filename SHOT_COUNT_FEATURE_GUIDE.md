# Guide: Adding 1 Shot / 5 Shot / 10 Shot Feature to ShotSync

This guide explains how to add the shot count selector (1 shot, 5 shots, 10 shots) to a ShotSync UI that doesn't have it yet.

## Overview

The feature allows users to select how many shots they want to analyze from their video. The system will:
- Show a UI selector with 1, 5, and 10 shot options
- Display a real-time shot counter overlay during processing
- Automatically stop processing when the target number of shots is reached
- Trim the video to only include the first N shots

---

## 1. CSS Styles

Add these styles to your `<style>` section (around line 112):

```css
/* Shot Count Selector Styles */
.shot-count-btn {
    padding: 10px 20px;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-card);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.3s;
    color: var(--text-dark);
}

.shot-count-btn:hover {
    border-color: var(--primary-color);
    background: rgba(255, 107, 122, 0.1);
}

.shot-count-btn.active {
    border-color: var(--primary-color);
    background: rgba(255, 107, 122, 0.15);
    color: var(--primary-color);
}
```

---

## 2. HTML UI Elements

### A. Shot Count Selector Buttons

Add this HTML in your recording controls section (around line 2484), typically near the record/upload buttons:

```html
<!-- Shot Count Selector -->
<div id="shotCountSelector" style="display: flex; flex-direction: column; gap: 10px; align-items: center; margin-bottom: 10px;">
    <label style="font-size: 14px; font-weight: 600; color: var(--text-dark);">Number of Shots to Analyze:</label>
    <div style="display: flex; gap: 10px;">
        <button class="shot-count-btn" data-count="1">1 Shot</button>
        <button class="shot-count-btn" data-count="5">5 Shots</button>
        <button class="shot-count-btn active" data-count="10">10 Shots</button>
    </div>
</div>
```

**Note:** The button with `class="active"` is the default selection (10 shots in this case).

### B. Shot Counter Overlay

Add this HTML in your video/skeleton viewer container (around line 2467). This displays the real-time shot count during processing:

```html
<!-- Translucent shot counter (positioned over video) -->
<div id="shotCounterOverlay" style="position: absolute; top: 20px; right: 20px; z-index: 25; background: rgba(0, 0, 0, 0.7); padding: 15px 25px; border-radius: 12px; border: 2px solid rgba(255, 107, 122, 0.5); display: none; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px);">
    <p style="margin: 0; font-size: 24px; font-weight: 700; color: var(--primary-color); text-align: center; font-family: 'Work Sans', sans-serif; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);">
        Shots Detected: <span id="shotCountValue" style="color: #fff;">0</span> / <span id="shotCountTarget" style="color: #fff;">10</span>
    </p>
</div>
```

**Important:** Make sure this is inside a container with `position: relative` so the absolute positioning works correctly.

---

## 3. JavaScript Variables

Add these variables in your main script section (around line 12423), typically near other loop/processing variables:

```javascript
let selectedShotCount = 10; // Default to 10 shots
let detectedShots = []; // Array to store detected shot sequences
```

**Note:** `selectedShotCount` should be accessible in the scope where you process video frames.

---

## 4. Event Handler for Shot Count Buttons

Add this event handler code (around line 10535). This uses an IIFE (Immediately Invoked Function Expression) to prevent duplicate event listeners:

```javascript
// Shot count selector handler - use direct event delegation (IIFE to avoid duplicates)
(function() {
    let handlerAttached = false;
    
    function attachHandler() {
        if (handlerAttached) return;
        handlerAttached = true;
        
        // Use a single event listener on the document
        document.addEventListener('click', function(e) {
            const btn = e.target.closest('.shot-count-btn');
            if (!btn) return;
            
            e.preventDefault();
            e.stopPropagation();
            
            console.log('Shot count button clicked:', btn.dataset.count);
            
            // Remove active class from all buttons
            document.querySelectorAll('.shot-count-btn').forEach(b => {
                b.classList.remove('active');
            });
            
            // Add active class to clicked button
            btn.classList.add('active');
            console.log('Active class added to button', btn.dataset.count);
            
            // Store selected count
            selectedShotCount = parseInt(btn.dataset.count);
            console.log('Selected shot count updated to:', selectedShotCount);
        }, true); // Use capture phase
    }
    
    // Attach immediately if DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachHandler);
    } else {
        attachHandler();
    }
})();
```

---

## 5. Initialize Shot Counter Overlay

In your video processing function (around line 13080), when you start processing, initialize and show the shot counter:

```javascript
// Get references to shot counter elements
const shotCountValue = document.getElementById('shotCountValue');
const shotCountTarget = document.getElementById('shotCountTarget');
const shotCounterOverlay = document.getElementById('shotCounterOverlay');

// Initialize and show translucent shot counter
if (shotCountTarget) {
    shotCountTarget.textContent = selectedShotCount;
}
if (shotCountValue) {
    shotCountValue.textContent = '0';
}
if (shotCounterOverlay) {
    shotCounterOverlay.style.display = 'block'; // Show translucent shot counter
}
```

---

## 6. Shot Detection Logic Integration

### A. Track Detected Shots

In your shot detection logic (around line 13280-13320), when a shot is detected, update the counter and check if target is reached:

```javascript
// When a shot is complete (e.g., wrist drops below shoulder after follow-through)
if (/* shot is complete condition */) {
    // Mark shot as detected
    shotDetectionState.shotsDetected++;
    
    // Store shot info
    const shotStartIndex = loop3DPoses.length - shotDetectionState.currentShotFrames.length;
    const shotEndIndex = loop3DPoses.length - 1;
    
    detectedShots.push({
        start: shotStartIndex,
        end: shotEndIndex,
        frames: shotDetectionState.currentShotFrames.length
    });
    
    console.log(`✓ Shot ${shotDetectionState.shotsDetected}/${selectedShotCount} complete`);
    
    // Update shot counter in real-time
    const shotCountValueEl = document.getElementById('shotCountValue');
    if (shotCountValueEl) {
        shotCountValueEl.textContent = shotDetectionState.shotsDetected;
    }
    
    // Check if we've reached the target
    if (shotDetectionState.shotsDetected >= selectedShotCount) {
        shotDetectionState.shouldStop = true;
        console.log(`Reached target of ${selectedShotCount} shots, will stop processing`);
    } else {
        // Reset and wait for next shot
        shotDetectionState.currentShotFrames = [];
        shotDetectionState.currentShotStages = [];
        shotDetectionState.currentShotAngles = [];
        shotDetectionState.state = "looking_for_pre_shot";
        shotDetectionState.seenFollowThrough = false;
        shotDetectionState.elbowDropped = false;
    }
}
```

### B. Trim to Selected Shot Count

After processing completes (around line 13445), if more shots were detected than requested, trim to only the first N shots:

```javascript
if (loop3DPoses.length > 0) {
    console.log(`Total shots detected: ${detectedShots.length}, total frames stored: ${loop3DPoses.length}`);
    
    // If we detected more shots than requested, trim to only the first N shots
    if (detectedShots.length > selectedShotCount) {
        // Calculate how many frames to keep (first N shots)
        let framesToKeep = 0;
        for (let i = 0; i < selectedShotCount; i++) {
            framesToKeep += detectedShots[i].frames;
        }
        
        // Trim arrays to only include first N shots
        loop3DPoses = loop3DPoses.slice(0, framesToKeep);
        loop3DStages = loop3DStages.slice(0, framesToKeep);
        loop3DAngles = loop3DAngles.slice(0, framesToKeep);
        
        console.log(`Trimmed to first ${selectedShotCount} shots: ${framesToKeep} frames`);
    }
    
    // Update status to show shot count
    if (statusEl) {
        const shotsUsed = Math.min(detectedShots.length, selectedShotCount);
        if (detectedShots.length >= selectedShotCount) {
            statusEl.textContent = `✓ Successfully detected and clipped ${shotsUsed} shot(s) (${loop3DPoses.length} frames)`;
        } else {
            statusEl.textContent = `✓ Detected ${detectedShots.length} shot(s), using all (${loop3DPoses.length} frames, requested: ${selectedShotCount})`;
        }
    }
    
    console.log(`Shot processing complete: ${Math.min(detectedShots.length, selectedShotCount)} shots, ${loop3DPoses.length} total frames`);
}
```

---

## 7. Integration Checklist

- [ ] Add CSS styles for `.shot-count-btn`
- [ ] Add HTML for shot count selector buttons
- [ ] Add HTML for shot counter overlay
- [ ] Add `selectedShotCount` and `detectedShots` variables
- [ ] Add event handler for shot count button clicks
- [ ] Initialize shot counter overlay when processing starts
- [ ] Update shot counter in real-time when shots are detected
- [ ] Check if target shot count is reached and stop processing
- [ ] Trim video data to only include first N shots after processing
- [ ] Update status message to show shot count information

---

## 8. Key Points

1. **Default Value:** The default is 10 shots (set by `selectedShotCount = 10` and the button with `class="active"`).

2. **Shot Detection:** The system detects shots by tracking pose stages (pre_shot → follow_through) and marking a shot complete when the wrist drops below the shoulder after follow-through.

3. **Real-time Updates:** The shot counter overlay updates in real-time as shots are detected.

4. **Automatic Trimming:** If more shots are detected than requested, only the first N shots are kept.

5. **Variable Scope:** Make sure `selectedShotCount` is accessible in both the event handler and the video processing function.

---

## 9. Testing

After adding all components:

1. Click different shot count buttons - verify the active state changes
2. Upload/record a video with multiple shots
3. Verify the shot counter overlay appears and updates in real-time
4. Verify processing stops when target shot count is reached
5. Verify only the requested number of shots are used in analysis

---

## 10. Troubleshooting

- **Buttons not responding:** Check that the event handler is attached and `selectedShotCount` variable is in the correct scope
- **Counter not showing:** Verify the overlay HTML is inside a container with `position: relative`
- **Counter not updating:** Check that `shotCountValue` element exists and is being updated in shot detection logic
- **Too many shots processed:** Verify the trimming logic runs after processing and checks `detectedShots.length > selectedShotCount`
