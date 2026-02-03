# Quick Reference: 1/5/10 Shot Feature Code Snippets

## All Code in One Place

### 1. CSS (add to `<style>` section)
```css
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

### 2. HTML - Shot Count Selector (add to recording controls)
```html
<div id="shotCountSelector" style="display: flex; flex-direction: column; gap: 10px; align-items: center; margin-bottom: 10px;">
    <label style="font-size: 14px; font-weight: 600; color: var(--text-dark);">Number of Shots to Analyze:</label>
    <div style="display: flex; gap: 10px;">
        <button class="shot-count-btn" data-count="1">1 Shot</button>
        <button class="shot-count-btn" data-count="5">5 Shots</button>
        <button class="shot-count-btn active" data-count="10">10 Shots</button>
    </div>
</div>
```

### 3. HTML - Shot Counter Overlay (add to video container)
```html
<div id="shotCounterOverlay" style="position: absolute; top: 20px; right: 20px; z-index: 25; background: rgba(0, 0, 0, 0.7); padding: 15px 25px; border-radius: 12px; border: 2px solid rgba(255, 107, 122, 0.5); display: none; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px);">
    <p style="margin: 0; font-size: 24px; font-weight: 700; color: var(--primary-color); text-align: center; font-family: 'Work Sans', sans-serif; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);">
        Shots Detected: <span id="shotCountValue" style="color: #fff;">0</span> / <span id="shotCountTarget" style="color: #fff;">10</span>
    </p>
</div>
```

### 4. JavaScript Variables (add near other processing variables)
```javascript
let selectedShotCount = 10; // Default to 10 shots
let detectedShots = []; // Array to store detected shot sequences
```

### 5. Event Handler (add to script section)
```javascript
// Shot count selector handler
(function() {
    let handlerAttached = false;
    function attachHandler() {
        if (handlerAttached) return;
        handlerAttached = true;
        document.addEventListener('click', function(e) {
            const btn = e.target.closest('.shot-count-btn');
            if (!btn) return;
            e.preventDefault();
            e.stopPropagation();
            document.querySelectorAll('.shot-count-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedShotCount = parseInt(btn.dataset.count);
            console.log('Selected shot count:', selectedShotCount);
        }, true);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachHandler);
    } else {
        attachHandler();
    }
})();
```

### 6. Initialize Counter (add when starting video processing)
```javascript
const shotCountValue = document.getElementById('shotCountValue');
const shotCountTarget = document.getElementById('shotCountTarget');
const shotCounterOverlay = document.getElementById('shotCounterOverlay');

if (shotCountTarget) shotCountTarget.textContent = selectedShotCount;
if (shotCountValue) shotCountValue.textContent = '0';
if (shotCounterOverlay) shotCounterOverlay.style.display = 'block';
```

### 7. Update Counter When Shot Detected (add in shot detection logic)
```javascript
// When shot is complete:
shotDetectionState.shotsDetected++;
detectedShots.push({
    start: shotStartIndex,
    end: shotEndIndex,
    frames: shotDetectionState.currentShotFrames.length
});

// Update UI
const shotCountValueEl = document.getElementById('shotCountValue');
if (shotCountValueEl) {
    shotCountValueEl.textContent = shotDetectionState.shotsDetected;
}

// Check if target reached
if (shotDetectionState.shotsDetected >= selectedShotCount) {
    shotDetectionState.shouldStop = true;
}
```

### 8. Trim to Selected Count (add after processing completes)
```javascript
if (detectedShots.length > selectedShotCount) {
    let framesToKeep = 0;
    for (let i = 0; i < selectedShotCount; i++) {
        framesToKeep += detectedShots[i].frames;
    }
    loop3DPoses = loop3DPoses.slice(0, framesToKeep);
    loop3DStages = loop3DStages.slice(0, framesToKeep);
    loop3DAngles = loop3DAngles.slice(0, framesToKeep);
}
```

---

## Integration Order

1. Add CSS styles
2. Add HTML for selector buttons
3. Add HTML for counter overlay
4. Add JavaScript variables
5. Add event handler
6. Initialize counter when processing starts
7. Update counter when shots detected
8. Trim to selected count after processing

---

## Key Variables to Track

- `selectedShotCount` - User's selected count (1, 5, or 10)
- `detectedShots` - Array of detected shot info
- `shotDetectionState.shotsDetected` - Current count of detected shots
- `shotDetectionState.shouldStop` - Flag to stop processing when target reached
