// ====================== GLOBAL STATE ======================
let benchmarkPoseData = [];
let userPoseData = [];
let benchmarkStream = null;
let userStream = null;
let benchmarkCamera = null;
let userCamera = null;
let comparisonChart = null;
let userInfo = null; // Store user info for email
let benchmarkRenderLoopId = null;
let userRenderLoopId = null;
let selectedPlayer = null; // 'curry', 'lebron', 'jordan', 'durant', 'clark', or 'custom'
let proPlayerBenchmarks = {}; // Store pre-loaded benchmarks for pro players
let benchmarkStopped = false; // Flag to prevent processing after stop

// EmailJS Configuration
// You'll get these from EmailJS dashboard after signing up
const EMAILJS_SERVICE_ID = 'YOUR_SERVICE_ID'; // Replace with your EmailJS service ID
const EMAILJS_TEMPLATE_ID = 'YOUR_TEMPLATE_ID'; // Replace with your EmailJS template ID
const EMAILJS_PUBLIC_KEY = 'YOUR_PUBLIC_KEY'; // Replace with your EmailJS public key

// EmailJS will be initialized when DOM loads

// Generate realistic example benchmark data for professional players
// Format must EXACTLY match what gets recorded in custom benchmark mode
function generateExampleBenchmarkData() {
    // Create a realistic shooting motion with typical angles
    // This mimics a real recording from startBenchmarkRecording
    const data = [];
    const duration = 2.0; // 2 seconds
    const fps = 90; // 90 frames per second  
    const totalFrames = Math.floor(duration * fps);
    
    let previousStage = "neutral";
    let recordingActive = false;
    let seenFollowThrough = false;
    let startTime = 0;
    
    for (let i = 0; i < totalFrames; i++) {
        const t = i / fps;
        const progress = t / duration; // 0 to 1
        
        // Realistic shooting motion states - match the state machine logic
        let state = "neutral";
        if (progress < 0.15) {
            state = "neutral";
        } else if (progress < 0.85) {
            state = "pre_shot";
            if (!recordingActive) {
                recordingActive = true;
                seenFollowThrough = false;
                startTime = t;
            }
        } else {
            state = "follow_through";
            if (recordingActive) {
                seenFollowThrough = true;
            }
        }
        
        // Elbow angle: starts at ~90 degrees, extends to ~180 degrees
        const elbowAngle = Math.max(0, Math.min(180, 90 + (progress * 90) + (Math.sin(progress * Math.PI) * 10)));
        
        // Wrist angle: starts at ~150 degrees, snaps to ~90 degrees at release
        const wristAngle = Math.max(0, Math.min(180, progress < 0.6 ? 150 : (150 - (progress - 0.6) * 150)));
        
        // Arm angle: relatively stable around 45-60 degrees
        const armAngle = Math.max(0, Math.min(180, 45 + (Math.sin(progress * Math.PI * 2) * 5)));
        
        // Ensure all angles are valid numbers (not NaN)
        if (isNaN(elbowAngle) || isNaN(wristAngle) || isNaN(armAngle)) {
            continue; // Skip invalid frames
        }
        
        // Create landmarks array - EXACT format: array of [x, y, z] arrays (like get3DPoint returns)
        const landmarks3D = [];
        for (let idx = 0; idx < 33; idx++) {
            // Create realistic landmark positions based on shooting motion
            const x = 0.5 + Math.sin(idx * 0.2) * 0.1 + (progress * 0.05);
            const y = 0.5 + Math.cos(idx * 0.2) * 0.1 - (progress * 0.1); // Arm moves up
            const z = Math.sin(idx * 0.15) * 0.05;
            landmarks3D.push([x, y, z]);
        }
        
        // Only record if recordingActive (matches the recording logic)
        if (recordingActive) {
            const elapsed = t - startTime;
            data.push({
                state: state,
                time: elapsed,  // Use elapsed time like in recording
                elbow_angle: elbowAngle,
                wrist_angle: wristAngle,
                arm_angle: armAngle,
                landmarks: landmarks3D
            });
        }
        
        previousStage = state;
    }
    
    return data;
}

// Initialize professional player benchmarks with example data
function initializeProPlayerBenchmarks() {
    const players = ['curry', 'lebron', 'jordan', 'durant', 'clark'];
    players.forEach(player => {
        // Use real data for LeBron if available, otherwise use synthetic
        if (player === 'lebron' && typeof window.lebron_benchmark_data !== 'undefined') {
            proPlayerBenchmarks[player] = window.lebron_benchmark_data;
            console.log(`Loaded real LeBron data: ${proPlayerBenchmarks[player].length} frames`);
        } else if (player === 'curry' && typeof window.curry_benchmark_data !== 'undefined') {
            proPlayerBenchmarks[player] = window.curry_benchmark_data;
            console.log(`Loaded real Curry data: ${proPlayerBenchmarks[player].length} frames`);
        } else {
        proPlayerBenchmarks[player] = generateExampleBenchmarkData();
        }
    });
}

// MediaPipe Pose
let benchmarkPose = null;
let userPose = null;

// ====================== MEDIAPIPE SETUP ======================

// POSE_CONNECTIONS - MediaPipe Pose landmark connections
const POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14],
    [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [11, 23], [12, 24], [23, 24],
    [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [27, 31], [28, 32],
    [29, 31], [30, 32]
];

// Helper function to get overlay color based on state
function getOverlayColor(state) {
    if (state === 'pre_shot') {
        return '#3b82f6'; // Blue
    } else if (state === 'follow_through') {
        return '#ff8c00'; // Orange
    } else {
        return '#00FF00'; // Green (neutral or default)
    }
}

// Drawing utility functions
function drawConnections(ctx, points, connections, style) {
    const { color = '#00FF00', lineWidth = 2 } = style || {};
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    
    const canvas = ctx.canvas;
    const width = canvas.width || canvas.offsetWidth || 640;
    const height = canvas.height || canvas.offsetHeight || 480;
    
    for (const [startIdx, endIdx] of connections) {
        const start = points[startIdx];
        const end = points[endIdx];
        if (start && end && start.visibility > 0.5 && end.visibility > 0.5) {
            ctx.moveTo(start.x * width, start.y * height);
            ctx.lineTo(end.x * width, end.y * height);
        }
    }
    
    ctx.stroke();
}

function drawLandmarks(ctx, points, style) {
    const { color = '#00FF00', lineWidth = 1, radius = 3 } = style || {};
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    
    const canvas = ctx.canvas;
    const width = canvas.width || canvas.offsetWidth || 640;
    const height = canvas.height || canvas.offsetHeight || 480;
    
    for (const point of points) {
        if (point && point.visibility > 0.5) {
            const x = point.x * width;
            const y = point.y * height;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

function initializePose() {
    // Initialize MediaPipe Pose
    const poseOptions = {
        modelComplexity: 2,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    };
    
    benchmarkPose = new Pose({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
        }
    });
    
    userPose = new Pose({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
        }
    });
    
    benchmarkPose.setOptions(poseOptions);
    userPose.setOptions(poseOptions);
}

// ====================== UTILITY FUNCTIONS ======================

function get3DPoint(landmarks, index, width, height) {
    if (!landmarks || index >= landmarks.length || landmarks[index].visibility < 0.5) {
        return null;
    }
    return [
        landmarks[index].x * width,
        landmarks[index].y * height,
        landmarks[index].z || 0
    ];
}

function calculateAngle(a, b, c) {
    if (!a || !b || !c) return null;
    
    const ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    
    const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    const magBA = Math.sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]);
    const magBC = Math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
    
    if (magBA < 1e-5 || magBC < 1e-5) return null;
    
    const cosine = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
    return Math.acos(cosine) * (180 / Math.PI);
}

/**
 * Normalize pose landmarks to align shoulders with x-axis.
 * This ensures consistent orientation regardless of camera angle.
 * 
 * @param {Array} landmarks - Array of 33 [x, y, z] points
 * @returns {Array} Normalized landmarks array
 */
function normalizePoseOrientation(landmarks) {
    if (!landmarks || landmarks.length < 33) {
        return landmarks;
    }
    
    // Get shoulder points (MediaPipe indices: 11 = left shoulder, 12 = right shoulder)
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    
    // Check if shoulders are valid
    if (!leftShoulder || !rightShoulder || 
        isNaN(leftShoulder[0]) || isNaN(rightShoulder[0])) {
        return landmarks; // Return original if shoulders not detected
    }
    
    // Calculate shoulder vector (from left to right shoulder)
    const shoulderVec = [
        rightShoulder[0] - leftShoulder[0],
        rightShoulder[1] - leftShoulder[1],
        rightShoulder[2] - leftShoulder[2]
    ];
    
    // Project to XZ plane (ignore Y for rotation calculation)
    const shoulderVecXZ = [shoulderVec[0], shoulderVec[2]];
    const shoulderMag = Math.sqrt(shoulderVecXZ[0] ** 2 + shoulderVecXZ[1] ** 2);
    
    if (shoulderMag < 1e-5) {
        return landmarks; // Shoulders too close, can't determine orientation
    }
    
    // Calculate angle to rotate shoulder vector to align with +X axis
    // Ensure the vector points in +X direction (right shoulder should have higher X than left)
    // Target: [1, 0] in XZ plane
    const targetVec = [1, 0];
    const currentVec = [shoulderVecXZ[0] / shoulderMag, shoulderVecXZ[1] / shoulderMag];
    
    // Calculate rotation angle (around Y-axis)
    const cosAngle = currentVec[0] * targetVec[0] + currentVec[1] * targetVec[1];
    const sinAngle = currentVec[0] * targetVec[1] - currentVec[1] * targetVec[0];
    let angle = Math.atan2(sinAngle, cosAngle);
    
    // If the dot product is negative, we need to flip direction (rotate 180 degrees)
    if (cosAngle < 0) {
        angle += Math.PI;
    }
    
    // Rotation matrix for rotation around Y-axis
    const cos = Math.cos(-angle); // Negative to align with +X
    const sin = Math.sin(-angle);
    
    // Calculate shoulder midpoint for translation
    const shoulderMidpoint = [
        (leftShoulder[0] + rightShoulder[0]) / 2,
        (leftShoulder[1] + rightShoulder[1]) / 2,
        (leftShoulder[2] + rightShoulder[2]) / 2
    ];
    
    // Apply rotation and translation to all landmarks
    const normalized = landmarks.map(landmark => {
        if (!landmark || isNaN(landmark[0])) {
            return [NaN, NaN, NaN];
        }
        
        // Translate to origin (using shoulder midpoint)
        const translated = [
            landmark[0] - shoulderMidpoint[0],
            landmark[1] - shoulderMidpoint[1],
            landmark[2] - shoulderMidpoint[2]
        ];
        
        // Rotate around Y-axis
        const rotated = [
            translated[0] * cos - translated[2] * sin,
            translated[1],
            translated[0] * sin + translated[2] * cos
        ];
        
        return rotated;
    });
    
    return normalized;
}

function getArmState(landmarks, width, height) {
    const rightShoulder = get3DPoint(landmarks, 12, width, height);
    const rightElbow = get3DPoint(landmarks, 14, width, height);
    const rightWrist = get3DPoint(landmarks, 16, width, height);
    const leftWrist = get3DPoint(landmarks, 15, width, height);
    const leftHip = get3DPoint(landmarks, 23, width, height);
    const rightHip = get3DPoint(landmarks, 24, width, height);

    if (rightWrist && leftWrist && leftHip && rightHip && rightShoulder) {
        const waistY = (leftHip[1] + rightHip[1]) / 2.0;
        const avgWristY = (rightWrist[1] + leftWrist[1]) / 2.0;
        const distWrists = Math.sqrt(
            Math.pow(rightWrist[0] - leftWrist[0], 2) +
            Math.pow(rightWrist[1] - leftWrist[1], 2) +
            Math.pow(rightWrist[2] - leftWrist[2], 2)
        );
        
        if (distWrists < 0.15 * width && avgWristY < waistY && rightWrist[1] > rightShoulder[1]) {
            return "pre_shot";
        }
    }

    if (rightWrist && rightShoulder) {
        if (rightShoulder[1] > rightWrist[1]) {
            return "follow_through";
        }
    }

    if (rightShoulder && rightWrist) {
        if (rightWrist[1] > rightShoulder[1]) {
            return "neutral";
        }
    }

    return "neutral";
}

// Enhanced stage detection for specific shooting phases
// Returns: "shot_start", "set_point", "follow_through", or null
function detectShootingStage(landmarks, width, height, currentState, previousState) {
    const rightShoulder = get3DPoint(landmarks, 12, width, height);
    const rightWrist = get3DPoint(landmarks, 16, width, height);
    
    if (!rightShoulder || !rightWrist) {
        return null;
    }
    
    // Shot Start: when shooting form begins (transition from neutral to pre_shot)
    if (previousState === "neutral" && currentState === "pre_shot") {
        return "shot_start";
    }
    
    // Set Point: when right wrist is higher than right shoulder (wrist Y < shoulder Y means higher on screen)
    if (currentState === "pre_shot" && rightWrist[1] < rightShoulder[1]) {
        return "set_point";
    }
    
    // Follow Through: when right wrist goes above right shoulder after set point
    // (This happens when we transition to follow_through state)
    if (currentState === "follow_through") {
        return "follow_through";
    }
    
    return null;
}

// Extract stage markers from pose data using exact get_arm_state logic
function extractStageMarkers(poseData) {
    const markers = {
        shot_start: null,
        set_point: null,
        follow_through: null
    };
    
    if (!poseData || poseData.length === 0) {
        return markers;
    }
    
    // Estimate coordinate system from landmark coordinates
    // Landmarks are normalized (translated/rotated) but still in pixel coordinate scale
    // Use a sample frame to estimate typical coordinate ranges
    let estimatedHeight = 480; // default
    let estimatedWidth = 640; // default
    let coordinateRange = 0;
    if (poseData.length > 0 && poseData[0].landmarks) {
        const sampleFrame = poseData[0];
        let maxY = 0, minY = Infinity, maxX = 0, minX = Infinity;
        for (let idx = 0; idx < sampleFrame.landmarks.length; idx++) {
            const landmark = sampleFrame.landmarks[idx];
            if (landmark && Array.isArray(landmark) && landmark.length >= 2 && !isNaN(landmark[0]) && !isNaN(landmark[1])) {
                maxX = Math.max(maxX, Math.abs(landmark[0]));
                minX = Math.min(minX, Math.abs(landmark[0]));
                maxY = Math.max(maxY, Math.abs(landmark[1]));
                minY = Math.min(minY, Math.abs(landmark[1]));
            }
        }
        // Calculate the range of coordinates
        coordinateRange = Math.max(maxY - minY, maxX - minX);
        // If coordinates are in pixel scale (normalized but still pixel-sized), estimate dimensions
        if (coordinateRange > 10) {
            // Likely pixel coordinates (normalized but still pixel scale)
            estimatedHeight = Math.max(estimatedHeight, coordinateRange * 2);
            estimatedWidth = Math.max(estimatedWidth, coordinateRange * 2);
        }
    }
    
    // Tolerance: Use a percentage of the coordinate range, with minimum threshold
    // For normalized pixel coordinates, use 1-2% of the coordinate range
    const tolerance = Math.max(5, coordinateRange * 0.02); // 2% of range, minimum 5 units
    
    let previousState = "neutral";
    let seenShotStart = false;
    
    // Track closest matches for debugging
    let closestSetPoint = { diff: Infinity, frame: null };
    let closestFollowThrough = { diff: Infinity, frame: null };
    
    for (let i = 0; i < poseData.length; i++) {
        const frame = poseData[i];
        if (!frame || !frame.landmarks) continue;
        
        // Use the state from frame (which should be determined by getArmState)
        // But if not present, we can determine it using getArmState logic
        let currentState = frame.state;
        if (!currentState && frame.landmarks) {
            // Re-determine state using getArmState logic (exact match to Python)
            // Note: landmarks are already in pixel coordinates from get3DPoint
            const rightShoulder = frame.landmarks[12];
            const rightWrist = frame.landmarks[16];
            const leftWrist = frame.landmarks[15];
            const leftHip = frame.landmarks[23];
            const rightHip = frame.landmarks[24];
            
            // Use exact get_arm_state logic from Python
            if (rightWrist && leftWrist && leftHip && rightHip && rightShoulder &&
                Array.isArray(rightWrist) && Array.isArray(leftWrist) && 
                Array.isArray(leftHip) && Array.isArray(rightHip) && Array.isArray(rightShoulder)) {
                const waistY = (leftHip[1] + rightHip[1]) / 2.0;
                const avgWristY = (rightWrist[1] + leftWrist[1]) / 2.0;
                const distWrists = Math.sqrt(
                    Math.pow(rightWrist[0] - leftWrist[0], 2) +
                    Math.pow(rightWrist[1] - leftWrist[1], 2) +
                    Math.pow(rightWrist[2] - leftWrist[2], 2)
                );
                if (distWrists < 0.15 * estimatedWidth && avgWristY < waistY && rightWrist[1] > rightShoulder[1]) {
                    currentState = "pre_shot";
                }
            }
            
            if (!currentState && rightWrist && rightShoulder && 
                Array.isArray(rightWrist) && Array.isArray(rightShoulder)) {
                if (rightShoulder[1] > rightWrist[1]) {
                    currentState = "follow_through";
                }
            }
            
            if (!currentState && rightShoulder && rightWrist &&
                Array.isArray(rightShoulder) && Array.isArray(rightWrist)) {
                if (rightWrist[1] > rightShoulder[1]) {
                    currentState = "neutral";
                }
            }
            
            if (!currentState) {
                currentState = "neutral";
            }
        }
        
        // Shot Start: first transition from neutral to pre_shot
        if (!markers.shot_start && previousState === "neutral" && currentState === "pre_shot") {
            markers.shot_start = {
                time: frame.time || 0,
                index: i,
                elbow_angle: frame.elbow_angle,
                wrist_angle: frame.wrist_angle,
                arm_angle: frame.arm_angle
            };
            seenShotStart = true;
        }
        
        // Set Point: when right wrist Y equals right shoulder Y (using exact equality check)
        // Track the FIRST occurrence during pre_shot phase
        if (!markers.set_point && currentState === "pre_shot" && seenShotStart) {
            const rightShoulder = frame.landmarks[12];
            const rightWrist = frame.landmarks[16];
            
            if (rightShoulder && rightWrist && 
                Array.isArray(rightShoulder) && Array.isArray(rightWrist)) {
                const shoulderY = rightShoulder[1];
                const wristY = rightWrist[1];
                
                if (!isNaN(shoulderY) && !isNaN(wristY)) {
                    // Check if wrist Y equals shoulder Y (within pixel tolerance)
                    const diff = Math.abs(wristY - shoulderY);
                    
                    // Track closest match for debugging
                    if (diff < closestSetPoint.diff) {
                        closestSetPoint.diff = diff;
                        closestSetPoint.frame = { time: frame.time, wristY, shoulderY, index: i };
                    }
                    
                    // Mark the FIRST occurrence when Y values are equal (within tolerance)
                    if (diff < tolerance && !markers.set_point) {
                        markers.set_point = {
                            time: frame.time || 0,
                            index: i,
                            elbow_angle: frame.elbow_angle,
                            wrist_angle: frame.wrist_angle,
                            arm_angle: frame.arm_angle,
                            wristY: wristY,
                            shoulderY: shoulderY
                        };
                        console.log('✅ Set Point detected at time:', frame.time, 'wristY:', wristY, 'shoulderY:', shoulderY, 'diff:', diff, 'tolerance:', tolerance);
                    }
                }
            }
        }
        
        // Follow Through: when right elbow Y equals right shoulder Y (using exact equality check)
        // Only detect after Set Point has been found - check ALL frames after set point
        // This ensures Set Point happens before Follow Through
        if (!markers.follow_through && markers.set_point && i > markers.set_point.index) {
            const rightShoulder = frame.landmarks[12];
            const rightElbow = frame.landmarks[14];
            
            if (rightShoulder && rightElbow &&
                Array.isArray(rightShoulder) && Array.isArray(rightElbow)) {
                const shoulderY = rightShoulder[1];
                const elbowY = rightElbow[1];
                
                if (!isNaN(shoulderY) && !isNaN(elbowY)) {
                    // Check if elbow Y equals shoulder Y (within pixel tolerance)
                    const diff = Math.abs(elbowY - shoulderY);
                    
                    // Track closest match for debugging (only after set point)
                    if (diff < closestFollowThrough.diff) {
                        closestFollowThrough.diff = diff;
                        closestFollowThrough.frame = { time: frame.time, elbowY, shoulderY, index: i };
                    }
                    
                    // Mark the FIRST occurrence after Set Point when Y values are equal (within tolerance)
                    if (diff < tolerance && !markers.follow_through) {
                        markers.follow_through = {
                            time: frame.time || 0,
                            index: i,
                            elbow_angle: frame.elbow_angle,
                            wrist_angle: frame.wrist_angle,
                            arm_angle: frame.arm_angle,
                            elbowY: elbowY,
                            shoulderY: shoulderY
                        };
                        console.log('✅ Follow Through detected at time:', frame.time, 'elbowY:', elbowY, 'shoulderY:', shoulderY, 'diff:', diff, 'tolerance:', tolerance, 'state:', currentState);
                    }
                }
            }
        }
        
        previousState = currentState;
    }
    
    console.log('Extracted stage markers:', markers);
    console.log('Closest Set Point match (even if not within tolerance):', closestSetPoint);
    console.log('Closest Follow Through match (even if not within tolerance):', closestFollowThrough);
    console.log('Used tolerance:', tolerance, 'coordinate range:', coordinateRange, 'estimated dimensions:', estimatedWidth, 'x', estimatedHeight);
    
    // Debug: Log sample Y values from different frames
    if (poseData.length > 0) {
        const sampleIndices = [0, Math.floor(poseData.length / 4), Math.floor(poseData.length / 2), Math.floor(poseData.length * 3 / 4), poseData.length - 1];
        console.log('Sample Y values across frames:');
        sampleIndices.forEach(idx => {
            if (idx < poseData.length && poseData[idx].landmarks) {
                const frame = poseData[idx];
                const rs = frame.landmarks[12];
                const rw = frame.landmarks[16];
                const re = frame.landmarks[14];
                if (rs && rw && re && Array.isArray(rs) && Array.isArray(rw) && Array.isArray(re)) {
                    console.log(`  Frame ${idx} (time: ${frame.time?.toFixed(2)}): Shoulder Y: ${rs[1].toFixed(2)}, Wrist Y: ${rw[1].toFixed(2)}, Elbow Y: ${re[1].toFixed(2)}, State: ${frame.state || 'unknown'}`);
                }
            }
        });
    }
    
    // If we didn't find markers but have close matches, use those with a more lenient threshold (3x tolerance)
    if (!markers.set_point && closestSetPoint.frame && closestSetPoint.diff < tolerance * 3) {
        const frame = poseData[closestSetPoint.frame.index];
        markers.set_point = {
            time: frame.time || 0,
            index: closestSetPoint.frame.index,
            elbow_angle: frame.elbow_angle,
            wrist_angle: frame.wrist_angle,
            arm_angle: frame.arm_angle,
            wristY: closestSetPoint.frame.wristY,
            shoulderY: closestSetPoint.frame.shoulderY
        };
        console.log('⚠️ Set Point found using closest match (diff:', closestSetPoint.diff.toFixed(2), ', tolerance:', (tolerance * 3).toFixed(2), ')');
    }
    
    // Only try to find Follow Through if Set Point was found, and only after Set Point chronologically
    if (!markers.follow_through && markers.set_point && closestFollowThrough.frame && closestFollowThrough.diff < tolerance * 3) {
        // Make sure Follow Through happens after Set Point
        if (closestFollowThrough.frame.index > markers.set_point.index) {
            const frame = poseData[closestFollowThrough.frame.index];
            markers.follow_through = {
                time: frame.time || 0,
                index: closestFollowThrough.frame.index,
                elbow_angle: frame.elbow_angle,
                wrist_angle: frame.wrist_angle,
                arm_angle: frame.arm_angle,
                elbowY: closestFollowThrough.frame.elbowY,
                shoulderY: closestFollowThrough.frame.shoulderY
            };
            console.log('⚠️ Follow Through found using closest match (diff:', closestFollowThrough.diff.toFixed(2), ', tolerance:', (tolerance * 3).toFixed(2), ')');
        } else {
            console.log('Follow Through closest match is before Set Point, skipping');
        }
    }
    
    return markers;
}

/**
 * Check if the full body is visible in the frame
 * Returns true if all key landmarks are visible with good confidence
 */
function isFullBodyVisible(landmarks) {
    if (!landmarks || landmarks.length < 33) {
        return false;
    }

    // Key landmarks to check for full body visibility
    const keyLandmarks = [
        0,  // nose
        11, // left shoulder
        12, // right shoulder
        23, // left hip
        24, // right hip
        25, // left knee
        26, // right knee
        27, // left ankle
        28  // right ankle
    ];

    const visibilityThreshold = 0.5;

    for (const index of keyLandmarks) {
        if (!landmarks[index] || landmarks[index].visibility < visibilityThreshold) {
            return false;
        }
    }

    return true;
}

// ====================== VIDEO CAPTURE ======================

async function startBenchmarkRecording() {
    try {
        const video = document.getElementById('benchmarkVideo');
        const canvas = document.getElementById('benchmarkOutput');
        const ctx = canvas.getContext('2d');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        benchmarkStream = stream;
        video.srcObject = stream;
        
        // Set canvas dimensions
        canvas.width = 640;
        canvas.height = 480;
        
        // Ensure video plays (non-blocking)
        video.play().catch(err => console.error('Video play error:', err));
        
        benchmarkPoseData = [];
        
        let previousStage = "neutral";
        let startTime = null;
        let recordingActive = false;
        let seenFollowThrough = false;
        const lastPrintTime = { value: Date.now() };
        
        document.getElementById('startBenchmark').disabled = true;
        document.getElementById('stopBenchmark').disabled = false;
        document.getElementById('benchmarkStatus').textContent = 'Recording...';
        document.getElementById('benchmarkStatus').className = 'status recording';
        
        benchmarkPose.onResults((results) => {
            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw the video frame
            if (results.image) {
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
            }

            // Check if full body is visible and show/hide warning
            const bodyWarning = document.getElementById('benchmarkBodyWarning');
            if (bodyWarning) {
                if (!results.poseLandmarks || !isFullBodyVisible(results.poseLandmarks)) {
                    bodyWarning.style.display = 'flex';
                } else {
                    bodyWarning.style.display = 'none';
                }
            }
            
            if (results.poseLandmarks) {
                const state = getArmState(results.poseLandmarks, canvas.width, canvas.height);
                const overlayColor = getOverlayColor(state);
                const currentTime = Date.now() / 1000.0;
                
                drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
                    color: overlayColor,
                    lineWidth: 2
                });
                drawLandmarks(ctx, results.poseLandmarks, {
                    color: overlayColor,
                    lineWidth: 1,
                    radius: 3
                });
                
                // Compute angles
                const rightShoulder = get3DPoint(results.poseLandmarks, 12, canvas.width, canvas.height);
                const rightElbow = get3DPoint(results.poseLandmarks, 14, canvas.width, canvas.height);
                const rightWrist = get3DPoint(results.poseLandmarks, 16, canvas.width, canvas.height);
                const rightIndex = get3DPoint(results.poseLandmarks, 20, canvas.width, canvas.height);
                const leftShoulder = get3DPoint(results.poseLandmarks, 11, canvas.width, canvas.height);
                
                const elbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
                const wristAngle = calculateAngle(rightElbow, rightWrist, rightIndex);
                const armAngle = calculateAngle(leftShoulder, rightShoulder, rightElbow);
                
                // Store landmarks
                const landmarks3D = [];
                for (let i = 0; i < 33; i++) {
                    const pt = get3DPoint(results.poseLandmarks, i, canvas.width, canvas.height);
                    landmarks3D.push(pt || [NaN, NaN, NaN]);
                }
                
                // Normalize pose orientation (align shoulders with x-axis)
                const normalizedLandmarks = normalizePoseOrientation(landmarks3D);
                
                // Stage transitions
                if (state !== previousStage) {
                    if (state === "pre_shot" && !recordingActive) {
                        recordingActive = true;
                        seenFollowThrough = false;
                        startTime = currentTime;
                        benchmarkPoseData = [];
                        lastPrintTime.value = currentTime;
                    } else if (state === "neutral" && recordingActive && !seenFollowThrough) {
                        recordingActive = false;
                        seenFollowThrough = false;
                        startTime = null;
                        benchmarkPoseData = [];
                    } else if (state === "follow_through" && recordingActive) {
                        seenFollowThrough = true;
                    } else if (state === "pre_shot" && recordingActive && seenFollowThrough) {
                        const elapsed = currentTime - startTime;
                        benchmarkPoseData.push({
                            state: state,
                            time: elapsed,
                            elbow_angle: elbowAngle,
                            wrist_angle: wristAngle,
                            arm_angle: armAngle,
                            landmarks: normalizedLandmarks
                        });
                        stopBenchmarkRecording();
                        return;
                    }
                    previousStage = state;
                }
                
                // Record while actively recording
                if (recordingActive) {
                    const elapsed = currentTime - startTime;
                    benchmarkPoseData.push({
                        state: state,
                        time: elapsed,
                        elbow_angle: elbowAngle,
                        wrist_angle: wristAngle,
                        arm_angle: armAngle,
                        landmarks: normalizedLandmarks
                    });
                    
                    if (state === "pre_shot" || state === "follow_through") {
                        if (currentTime - lastPrintTime.value >= 0.1) {
                            lastPrintTime.value = currentTime;
                        }
                    }
                }
            }
            
            ctx.restore();
        });
        
        benchmarkCamera = new Camera(video, {
            onFrame: async () => {
                await benchmarkPose.send({image: video});
            },
            width: 640,
            height: 480
        });
        benchmarkCamera.start();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        document.getElementById('benchmarkStatus').textContent = 'Error accessing camera. Please allow camera permissions.';
        document.getElementById('benchmarkStatus').className = 'status error';
    }
}

function stopBenchmarkRecording() {
    if (benchmarkCamera) {
        benchmarkCamera.stop();
        benchmarkCamera = null;
    }
    
    if (benchmarkStream) {
        benchmarkStream.getTracks().forEach(track => track.stop());
        benchmarkStream = null;
    }
    
    document.getElementById('startBenchmark').disabled = false;
    document.getElementById('stopBenchmark').disabled = true;
    
    if (benchmarkPoseData.length > 0) {
        document.getElementById('benchmarkStatus').textContent = `Recorded ${benchmarkPoseData.length} frames.`;
        document.getElementById('benchmarkStatus').className = 'status success';
        document.getElementById('retakeBenchmark').style.display = 'inline-block';
        
        // Move to step 2
        document.getElementById('step1').classList.remove('active');
        document.getElementById('step1').style.display = 'none';
        document.getElementById('step2').classList.add('active');
        document.getElementById('step2').style.display = 'block';
        
        // Update back button visibility - no longer needed as buttons are visible by default
        // const backBtn = document.getElementById('backToPlayers');
        // if (backBtn) backBtn.style.display = 'block';
    }
}

async function startUserRecording() {
    try {
        const video = document.getElementById('userVideo');
        const canvas = document.getElementById('userOutput');
        const ctx = canvas.getContext('2d');
        
        // Set canvas dimensions
        canvas.width = 640;
        canvas.height = 480;

        // Get webcam stream
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        userStream = stream;
        video.srcObject = stream;
        userPoseData = [];
        
        let previousStage = "neutral";
        let startTime = null;
        let recordingActive = false;
        let seenFollowThrough = false;
        const lastPrintTime = { value: Date.now() };
        
        document.getElementById('startUser').disabled = true;
        document.getElementById('stopUser').disabled = false;
        document.getElementById('userStatus').textContent = 'Processing video...';
        document.getElementById('userStatus').className = 'status recording';
        
        userPose.onResults((results) => {
            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw the video frame
            if (results.image) {
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
            }

            // Check if full body is visible and show/hide warning
            const bodyWarning = document.getElementById('userBodyWarning');
            if (bodyWarning) {
                if (!results.poseLandmarks || !isFullBodyVisible(results.poseLandmarks)) {
                    bodyWarning.style.display = 'flex';
                } else {
                    bodyWarning.style.display = 'none';
                }
            }
            
            if (results.poseLandmarks) {
                const state = getArmState(results.poseLandmarks, canvas.width, canvas.height);
                const overlayColor = getOverlayColor(state);
                const currentTime = Date.now() / 1000.0;
                
                drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
                    color: overlayColor,
                    lineWidth: 2
                });
                drawLandmarks(ctx, results.poseLandmarks, {
                    color: overlayColor,
                    lineWidth: 1,
                    radius: 3
                });
                
                const rightShoulder = get3DPoint(results.poseLandmarks, 12, canvas.width, canvas.height);
                const rightElbow = get3DPoint(results.poseLandmarks, 14, canvas.width, canvas.height);
                const rightWrist = get3DPoint(results.poseLandmarks, 16, canvas.width, canvas.height);
                const rightIndex = get3DPoint(results.poseLandmarks, 20, canvas.width, canvas.height);
                const leftShoulder = get3DPoint(results.poseLandmarks, 11, canvas.width, canvas.height);
                
                const elbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
                const wristAngle = calculateAngle(rightElbow, rightWrist, rightIndex);
                const armAngle = calculateAngle(leftShoulder, rightShoulder, rightElbow);
                
                const landmarks3D = [];
                for (let i = 0; i < 33; i++) {
                    const pt = get3DPoint(results.poseLandmarks, i, canvas.width, canvas.height);
                    landmarks3D.push(pt || [NaN, NaN, NaN]);
                }
                
                // Normalize pose orientation (align shoulders with x-axis)
                const normalizedLandmarks = normalizePoseOrientation(landmarks3D);
                
                if (state !== previousStage) {
                    if (state === "pre_shot" && !recordingActive) {
                        recordingActive = true;
                        seenFollowThrough = false;
                        startTime = currentTime;
                        userPoseData = [];
                        lastPrintTime.value = currentTime;
                        document.getElementById('userStatus').textContent = 'Recording shot...';
                        console.log('Shot detected! Recording started.');
                    } else if (state === "neutral" && recordingActive && !seenFollowThrough) {
                        recordingActive = false;
                        seenFollowThrough = false;
                        startTime = null;
                        userPoseData = [];
                    } else if (state === "follow_through" && recordingActive) {
                        seenFollowThrough = true;
                    } else if (state === "pre_shot" && recordingActive && seenFollowThrough) {
                        const elapsed = currentTime - startTime;
                        userPoseData.push({
                            state: state,
                            time: elapsed,
                            elbow_angle: elbowAngle,
                            wrist_angle: wristAngle,
                            arm_angle: armAngle,
                            landmarks: normalizedLandmarks
                        });
                        stopUserRecording();
                        return;
                    }
                    previousStage = state;
                }
                
                if (recordingActive) {
                    const elapsed = currentTime - startTime;
                    userPoseData.push({
                        state: state,
                        time: elapsed,
                        elbow_angle: elbowAngle,
                        wrist_angle: wristAngle,
                        arm_angle: armAngle,
                        landmarks: normalizedLandmarks
                    });
                    
                    if (state === "pre_shot" || state === "follow_through") {
                        if (currentTime - lastPrintTime.value >= 0.1) {
                            lastPrintTime.value = currentTime;
                        }
                    }
                }
            }
            
            ctx.restore();
        });
        
        // Use Camera utility to process frames
        userCamera = new Camera(video, {
            onFrame: async () => {
                await userPose.send({image: video});
            },
            width: 640,
            height: 480
        });
        userCamera.start();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        document.getElementById('userStatus').textContent = 'Error accessing camera. Please allow camera permissions.';
        document.getElementById('userStatus').className = 'status error';
    }
}

function stopUserRecording() {
    if (userCamera) {
        userCamera.stop();
        userCamera = null;
    }
    
    if (userStream) {
        userStream.getTracks().forEach(track => track.stop());
        userStream = null;
    }
    
    document.getElementById('startUser').disabled = false;
    document.getElementById('stopUser').disabled = true;
    
    if (userPoseData.length > 0) {
        document.getElementById('userStatus').textContent = `Recorded ${userPoseData.length} frames. Analyzing...`;
        document.getElementById('userStatus').className = 'status success';
        document.getElementById('retakeUser').style.display = 'inline-block';
        
        compareShots();
    }
}

async function processUploadedUserVideo() {
    try {
        const video = document.getElementById('userVideo');
        const canvas = document.getElementById('userOutput');
        const ctx = canvas.getContext('2d');
        const statusEl = document.getElementById('userStatus');

        // Disable process button
        document.getElementById('processUserVideo').disabled = true;
        document.getElementById('processUserVideo').textContent = 'Processing...';

        // Set canvas dimensions
        canvas.width = 640;
        canvas.height = 480;

        userPoseData = [];
        let previousStage = "neutral";
        let startTime = null;
        let recordingActive = false;
        let seenFollowThrough = false;

        statusEl.textContent = 'Processing video...';
        statusEl.className = 'status recording';

        // Wait for video to be ready
        await new Promise((resolve) => {
            if (video.readyState >= 2) {
                resolve();
            } else {
                video.addEventListener('loadeddata', resolve, { once: true });
            }
        });

        // Reset video to beginning
        video.currentTime = 0;
        await video.play();

        // Process each frame
        const processFrame = async () => {
            return new Promise((resolve) => {
                userPose.onResults((results) => {
                    ctx.save();
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // Draw the video frame
                    if (results.image) {
                        ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
                    }

                    // Check if full body is visible and show/hide warning
                    const bodyWarning = document.getElementById('userBodyWarning');
                    if (bodyWarning) {
                        if (!results.poseLandmarks || !isFullBodyVisible(results.poseLandmarks)) {
                            bodyWarning.style.display = 'flex';
                        } else {
                            bodyWarning.style.display = 'none';
                        }
                    }

                    if (results.poseLandmarks) {
                        const state = getArmState(results.poseLandmarks, canvas.width, canvas.height);
                        const overlayColor = getOverlayColor(state);
                        const currentTime = video.currentTime;
                        
                        drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
                            color: overlayColor,
                            lineWidth: 2
                        });
                        drawLandmarks(ctx, results.poseLandmarks, {
                            color: overlayColor,
                            lineWidth: 1,
                            radius: 3
                        });

                        const rightShoulder = get3DPoint(results.poseLandmarks, 12, canvas.width, canvas.height);
                        const rightElbow = get3DPoint(results.poseLandmarks, 14, canvas.width, canvas.height);
                        const rightWrist = get3DPoint(results.poseLandmarks, 16, canvas.width, canvas.height);
                        const rightIndex = get3DPoint(results.poseLandmarks, 20, canvas.width, canvas.height);
                        const leftShoulder = get3DPoint(results.poseLandmarks, 11, canvas.width, canvas.height);

                        const elbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
                        const wristAngle = calculateAngle(rightElbow, rightWrist, rightIndex);
                        const armAngle = calculateAngle(leftShoulder, rightShoulder, rightElbow);

                        const landmarks3D = [];
                        for (let i = 0; i < 33; i++) {
                            const pt = get3DPoint(results.poseLandmarks, i, canvas.width, canvas.height);
                            landmarks3D.push(pt || [NaN, NaN, NaN]);
                        }

                        // Normalize pose orientation
                        const normalizedLandmarks = normalizePoseOrientation(landmarks3D);

                        if (state !== previousStage) {
                            if (state === "pre_shot" && !recordingActive) {
                                recordingActive = true;
                                seenFollowThrough = false;
                                startTime = currentTime;
                                userPoseData = [];
                                statusEl.textContent = 'Shot detected! Processing...';
                            } else if (state === "neutral" && recordingActive && !seenFollowThrough) {
                                recordingActive = false;
                                seenFollowThrough = false;
                                startTime = null;
                                userPoseData = [];
                            } else if (state === "follow_through" && recordingActive) {
                                seenFollowThrough = true;
                            } else if (state === "pre_shot" && recordingActive && seenFollowThrough) {
                                const elapsed = currentTime - startTime;
                                userPoseData.push({
                                    state: state,
                                    time: elapsed,
                                    elbow_angle: elbowAngle,
                                    wrist_angle: wristAngle,
                                    arm_angle: armAngle,
                                    landmarks: normalizedLandmarks
                                });
                                resolve(true); // Shot complete
                                return;
                            }
                            previousStage = state;
                        }

                        if (recordingActive) {
                            const elapsed = currentTime - startTime;
                            userPoseData.push({
                                state: state,
                                time: elapsed,
                                elbow_angle: elbowAngle,
                                wrist_angle: wristAngle,
                                arm_angle: armAngle,
                                landmarks: normalizedLandmarks
                            });
                        }
                    }

                    ctx.restore();
                    resolve(false); // Continue processing
                });
            });
        };

        // Process video frame by frame
        const frameInterval = 1 / 30; // 30 FPS
        while (video.currentTime < video.duration) {
            await userPose.send({ image: video });
            const shotComplete = await processFrame();

            if (shotComplete) {
                break;
            }

            // Advance to next frame
            video.currentTime += frameInterval;
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        video.pause();

        // Check if we captured any data
        if (userPoseData.length > 0) {
            statusEl.textContent = `Processed ${userPoseData.length} frames. Analyzing...`;
            statusEl.className = 'status success';
            document.getElementById('retakeUser').style.display = 'inline-block';

            compareShots();
        } else {
            statusEl.textContent = 'No shot detected in video. Please try another video or record live.';
            statusEl.className = 'status error';

            document.getElementById('processUserVideo').disabled = false;
            document.getElementById('processUserVideo').textContent = 'Analyze Video';
        }

    } catch (error) {
        console.error('Error processing video:', error);
        document.getElementById('userStatus').textContent = 'Error processing video. Please try again.';
        document.getElementById('userStatus').className = 'status error';

        document.getElementById('processUserVideo').disabled = false;
        document.getElementById('processUserVideo').textContent = 'Analyze Video';
    }
}

/**
 * Process a video file to extract pose data and generate benchmark data
 * This function can be used to process Curry's shot video and generate curry_benchmark.js
 * Usage: processVideoForBenchmark(videoFile, 'curry')
 */
async function processVideoForBenchmark(videoFile, playerName = 'curry') {
    return new Promise(async (resolve, reject) => {
        try {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(videoFile);
            video.muted = true;
            video.playsInline = true;
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            const poseData = [];
            let previousStage = "neutral";
            let startTime = null;
            let recordingActive = false;
            let seenFollowThrough = false;
            
            // Wait for video to load
            await new Promise((resolve) => {
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    resolve();
                }, { once: true });
            });
            
            await video.play();
            
            // Create a temporary pose instance for processing
            const tempPose = new Pose({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
                }
            });
            
            tempPose.setOptions({
                modelComplexity: 2,
                smoothLandmarks: true,
                enableSegmentation: false,
                smoothSegmentation: false,
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.7
            });
            
            const processFrame = async () => {
                return new Promise((resolve) => {
                    tempPose.onResults((results) => {
                        ctx.save();
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        
                        // Draw the video frame
                        if (results.image) {
                            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
                        }
                        
                        if (results.poseLandmarks) {
                            const state = getArmState(results.poseLandmarks, canvas.width, canvas.height);
                            const overlayColor = getOverlayColor(state);
                            const currentTime = video.currentTime;
                            
                            // Draw overlay
                            drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
                                color: overlayColor,
                                lineWidth: 2
                            });
                            drawLandmarks(ctx, results.poseLandmarks, {
                                color: overlayColor,
                                lineWidth: 1,
                                radius: 3
                            });
                            
                            const rightShoulder = get3DPoint(results.poseLandmarks, 12, canvas.width, canvas.height);
                            const rightElbow = get3DPoint(results.poseLandmarks, 14, canvas.width, canvas.height);
                            const rightWrist = get3DPoint(results.poseLandmarks, 16, canvas.width, canvas.height);
                            const rightIndex = get3DPoint(results.poseLandmarks, 20, canvas.width, canvas.height);
                            const leftShoulder = get3DPoint(results.poseLandmarks, 11, canvas.width, canvas.height);
                            
                            const elbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
                            const wristAngle = calculateAngle(rightElbow, rightWrist, rightIndex);
                            const armAngle = calculateAngle(leftShoulder, rightShoulder, rightElbow);
                            
                            const landmarks3D = [];
                            for (let i = 0; i < 33; i++) {
                                const pt = get3DPoint(results.poseLandmarks, i, canvas.width, canvas.height);
                                landmarks3D.push(pt || [NaN, NaN, NaN]);
                            }
                            
                            const normalizedLandmarks = normalizePoseOrientation(landmarks3D);
                            
                            // Stage transitions
                            if (state !== previousStage) {
                                if (state === "pre_shot" && !recordingActive) {
                                    recordingActive = true;
                                    seenFollowThrough = false;
                                    startTime = currentTime;
                                    poseData.length = 0; // Clear previous data
                                } else if (state === "neutral" && recordingActive && !seenFollowThrough) {
                                    recordingActive = false;
                                    seenFollowThrough = false;
                                    startTime = null;
                                    poseData.length = 0;
                                } else if (state === "follow_through" && recordingActive) {
                                    seenFollowThrough = true;
                                } else if (state === "pre_shot" && recordingActive && seenFollowThrough) {
                                    const elapsed = currentTime - startTime;
                                    poseData.push({
                                        state: state,
                                        time: elapsed,
                                        elbow_angle: elbowAngle,
                                        wrist_angle: wristAngle,
                                        arm_angle: armAngle,
                                        landmarks: normalizedLandmarks
                                    });
                                    resolve(true); // Shot complete
                                    return;
                                }
                                previousStage = state;
                            }
                            
                            // Record while actively recording
                            if (recordingActive) {
                                const elapsed = currentTime - startTime;
                                poseData.push({
                                    state: state,
                                    time: elapsed,
                                    elbow_angle: elbowAngle,
                                    wrist_angle: wristAngle,
                                    arm_angle: armAngle,
                                    landmarks: normalizedLandmarks
                                });
                            }
                        }
                        
                        ctx.restore();
                        resolve(false); // Continue processing
                    });
                });
            };
            
            // Process video frame by frame
            const frameInterval = 1 / 30; // 30 FPS
            while (video.currentTime < video.duration) {
                await tempPose.send({ image: video });
                const shotComplete = await processFrame();
                
                if (shotComplete) {
                    break;
                }
                
                // Advance to next frame
                video.currentTime += frameInterval;
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            video.pause();
            URL.revokeObjectURL(video.src);
            
            if (poseData.length > 0) {
                // Generate the benchmark file content
                const benchmarkContent = `// ${playerName.toUpperCase()} benchmark data (extracted from video)
const ${playerName}_data = ${JSON.stringify(poseData, null, 2)};
`;
                
                // Create download link
                const blob = new Blob([benchmarkContent], { type: 'text/javascript' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${playerName}_benchmark.js`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                console.log(`Generated ${playerName}_benchmark.js with ${poseData.length} frames`);
                resolve(poseData);
            } else {
                reject(new Error('No shot detected in video'));
            }
            
        } catch (error) {
            console.error('Error processing video for benchmark:', error);
            reject(error);
        }
    });
}

// Make function available globally for console usage
window.processVideoForBenchmark = processVideoForBenchmark;

// ====================== SHOT COMPARISON ======================

function computeOverallForm(e, w, a) {
    const angles = [];
    if (e !== null && e !== undefined) angles.push(e);
    if (w !== null && w !== undefined) angles.push(w);
    if (a !== null && a !== undefined) angles.push(a);
    if (angles.length === 0) return null;
    return angles.reduce((a, b) => a + b, 0) / angles.length;
}

/**
 * Extract a single-dimensional form series from shot data.
 * Uses a weighted combination of key angles for better comparison.
 */
function extractFormSeries(shotData) {
    const times = [];
    const formVals = [];
    for (const entry of shotData) {
        // Use weighted combination: elbow angle is most important for shooting form
        let measure = null;
        if (entry.elbow_angle !== null && entry.elbow_angle !== undefined) {
            measure = entry.elbow_angle * 0.5; // 50% weight
            if (entry.wrist_angle !== null && entry.wrist_angle !== undefined) {
                measure += entry.wrist_angle * 0.3; // 30% weight
            }
            if (entry.arm_angle !== null && entry.arm_angle !== undefined) {
                measure += entry.arm_angle * 0.2; // 20% weight
            }
        } else if (entry.wrist_angle !== null && entry.wrist_angle !== undefined) {
            measure = entry.wrist_angle;
        } else if (entry.arm_angle !== null && entry.arm_angle !== undefined) {
            measure = entry.arm_angle;
        }
        
        if (measure !== null) {
            times.push(entry.time);
            formVals.push(measure);
        }
    }
    return { times, formVals };
}

// Simple DTW implementation
function dtw(series1, series2) {
    const n = series1.length;
    const m = series2.length;
    const dtwMatrix = Array(n + 1).fill(null).map(() => Array(m + 1).fill(Infinity));
    dtwMatrix[0][0] = 0;
    
    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= m; j++) {
            const cost = Math.abs(series1[i - 1] - series2[j - 1]);
            dtwMatrix[i][j] = cost + Math.min(
                dtwMatrix[i - 1][j],
                dtwMatrix[i][j - 1],
                dtwMatrix[i - 1][j - 1]
            );
        }
    }
    
    // Build path
    const path = [];
    let i = n, j = m;
    while (i > 0 && j > 0) {
        path.unshift([i - 1, j - 1]);
        const prev = [
            dtwMatrix[i - 1][j],
            dtwMatrix[i][j - 1],
            dtwMatrix[i - 1][j - 1]
        ];
        const minIdx = prev.indexOf(Math.min(...prev));
        if (minIdx === 0) i--;
        else if (minIdx === 1) j--;
        else { i--; j--; }
    }
    
    return { distance: dtwMatrix[n][m], path };
}

/**
 * Compute user closeness scores based on angle differences.
 * More accurate: uses a steeper penalty curve for better discrimination.
 */
function computeUserCloseness(benchForm, userForm, path) {
    // More accurate alpha: for angles (0-180°), a 30° difference should be significant
    // Formula: 100 - (diff / maxDiff) * 100, where maxDiff = 30° for 0% similarity
    const maxAngleDiff = 30.0; // 30° difference = 0% similarity
    const userMap = {};
    
    for (const [i, j] of path) {
        if (!userMap[j]) userMap[j] = [];
        userMap[j].push(i);
    }
    
    const userCloseness = [];
    for (let j = 0; j < userForm.length; j++) {
        if (userMap[j]) {
            const iList = userMap[j];
            const iMid = iList[Math.floor(iList.length / 2)];
            const diff = Math.abs(userForm[j] - benchForm[iMid]);
            // Linear scaling: 0° diff = 100%, 30° diff = 0%
            const score = Math.max(0, Math.min(100, 100 - (diff / maxAngleDiff) * 100));
            userCloseness.push(score);
        } else {
            // If no match found, give a neutral score (not 100%)
            userCloseness.push(50);
        }
    }
    return userCloseness;
}

async function compareShots() {
    document.getElementById('step2').classList.remove('active');
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step3').classList.add('active');
    document.getElementById('step3').style.display = 'block';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    setTimeout(async () => {
        // For pro players, use pre-loaded benchmark; for custom, use recorded benchmark
        let benchmarkData = benchmarkPoseData;
        
        if (selectedPlayer && selectedPlayer !== 'custom') {
            // Use pre-loaded benchmark for pro player
            if (proPlayerBenchmarks[selectedPlayer] && proPlayerBenchmarks[selectedPlayer].length > 0) {
                benchmarkData = proPlayerBenchmarks[selectedPlayer];
            } else {
                // Initialize if needed
                initializeProPlayerBenchmarks();
                benchmarkData = proPlayerBenchmarks[selectedPlayer];
            }
        }
        
        // Check if we have valid benchmark data
        if (!benchmarkData || benchmarkData.length === 0) {
            document.getElementById('loading').innerHTML = '<p style="color: red;">No benchmark data found. Please record a benchmark first.</p>';
            return;
        }
        
        if (userPoseData.length === 0) {
            document.getElementById('loading').innerHTML = '<p style="color: red;">No user shot data found. Please record your shot again.</p>';
            return;
        }
        
        const benchForm = extractFormSeries(benchmarkData);
        const userForm = extractFormSeries(userPoseData);
        
        if (benchForm.times.length < 2 || userForm.times.length < 2) {
            document.getElementById('loading').innerHTML = '<p style="color: red;">Insufficient data for comparison. Please record again.</p>';
            return;
        }
        
        const { distance, path } = dtw(benchForm.formVals, userForm.formVals);
        const userCloseness = computeUserCloseness(benchForm.formVals, userForm.formVals, path);
        
        const avgCloseness = userCloseness.reduce((a, b) => a + b, 0) / userCloseness.length;
        
        // Save similarity score to training database
        if (window.saveSimilarityScore && window.firebaseAuth?.currentUser) {
            try {
                const userId = window.firebaseAuth.currentUser.uid;
                const player = selectedPlayer || 'custom';
                await window.saveSimilarityScore(userId, player, avgCloseness);
                
                // Check and award similarity badges
                if (avgCloseness >= 80 && avgCloseness < 90) {
                    await window.awardBadge(userId, 'similarity', 'bronze');
                } else if (avgCloseness >= 90 && avgCloseness < 95) {
                    await window.awardBadge(userId, 'similarity', 'silver');
                } else if (avgCloseness >= 95) {
                    await window.awardBadge(userId, 'similarity', 'gold');
                }
                
                // Check all other badges after saving score
                if (window.checkAllBadges) {
                    await window.checkAllBadges(userId);
                }
            } catch (error) {
                console.error('Error saving similarity score:', error);
            }
        }
        
        // Extract stage markers for both benchmark and user
        const benchStageMarkers = extractStageMarkers(benchmarkData);
        const userStageMarkers = extractStageMarkers(userPoseData);
        
        // Generate feedback
        const feedback = [];
        feedback.push(`Overall Score: ${avgCloseness.toFixed(1)}%`);
        
        if (avgCloseness >= 90) {
            feedback.push("Excellent form! Your shot closely matches the benchmark.");
        } else if (avgCloseness >= 75) {
            feedback.push("Good form with room for improvement.");
        } else if (avgCloseness >= 60) {
            feedback.push("Your form needs work. Focus on key areas.");
        } else {
            feedback.push("Significant differences detected. Review the feedback below.");
        }
        
        displayResults({
            benchTimes: benchForm.times,
            userTimes: userForm.times,
            userCloseness: userCloseness,
            feedback: feedback,
            playerName: selectedPlayer,
            benchStageMarkers: benchStageMarkers,
            userStageMarkers: userStageMarkers,
            benchmarkData: benchmarkData,
            userPoseData: userPoseData
        });
    }, 500);
}

// Helper function to format coordinate to 3 significant figures
function formatCoordinate(value) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }
    // Convert to 3 significant figures
    const magnitude = Math.abs(value);
    if (magnitude === 0) return '0';
    
    const order = Math.floor(Math.log10(magnitude));
    const factor = Math.pow(10, 2 - order);
    const rounded = Math.round(value * factor) / factor;
    
    // Format to show 3 significant figures
    return rounded.toPrecision(3);
}

// Helper function to format coordinates as (x,y,z)
function formatCoordinates(landmark) {
    if (!landmark || !Array.isArray(landmark) || landmark.length < 3) {
        return '(N/A, N/A, N/A)';
    }
    return `(${formatCoordinate(landmark[0])}, ${formatCoordinate(landmark[1])}, ${formatCoordinate(landmark[2])})`;
}

// Populate coordinate table with benchmark and user coordinates
function populateCoordinateTable(data) {
    console.log('populateCoordinateTable called with data:', data);
    
    // Initialize transitions object (always return this)
    const transitions = {
        benchSetPoint: [],
        benchFollowThrough: [],
        benchShotEnd: [],
        userSetPoint: [],
        userFollowThrough: [],
        userShotEnd: []
    };
    
    const tableBody = document.getElementById('coordinateTableBody');
    if (!tableBody) {
        console.warn('Coordinate table body not found');
        return transitions; // Return empty transitions instead of undefined
    }
    
    // Clear existing rows
    tableBody.innerHTML = '';
    
    // Get pose data - check both possible property names
    const benchmarkPoseData = data.benchmarkData || data.benchmarkPoseData || [];
    const userPoseData = data.userPoseData || [];
    const userTimes = data.userTimes || [];
    const benchTimes = data.benchTimes || data.userTimes || [];
    
    console.log('Benchmark pose data length:', benchmarkPoseData.length);
    console.log('User pose data length:', userPoseData.length);
    console.log('User times length:', userTimes.length);
    
    if (userTimes.length === 0) {
        console.warn('No user times available for table');
        return transitions; // Return empty transitions instead of undefined
    }
    
    // Get stage markers for highlighting
    const benchSetPoint = data.benchStageMarkers?.set_point;
    const benchFollowThrough = data.benchStageMarkers?.follow_through;
    const userSetPoint = data.userStageMarkers?.set_point;
    const userFollowThrough = data.userStageMarkers?.follow_through;
    
    // Create a map of time -> benchmark pose data index
    const benchTimeMap = new Map();
    benchmarkPoseData.forEach((frame, idx) => {
        if (frame && frame.time !== undefined && frame.time !== null) {
            benchTimeMap.set(frame.time, idx);
        }
    });
    
    // Create a map of time -> user pose data index
    const userTimeMap = new Map();
    userPoseData.forEach((frame, idx) => {
        if (frame && frame.time !== undefined && frame.time !== null) {
            userTimeMap.set(frame.time, idx);
        }
    });
    
    // Find the maximum time from both datasets
    let maxTime = 0;
    
    // Get max time from benchmark
    benchmarkPoseData.forEach((frame) => {
        if (frame && frame.time !== undefined && frame.time !== null) {
            maxTime = Math.max(maxTime, frame.time);
        }
    });
    
    // Get max time from user
    userPoseData.forEach((frame) => {
        if (frame && frame.time !== undefined && frame.time !== null) {
            maxTime = Math.max(maxTime, frame.time);
        }
    });
    
    // Also check userTimes
    if (userTimes.length > 0) {
        userTimes.forEach(time => {
            if (time !== undefined && time !== null) {
                maxTime = Math.max(maxTime, time);
            }
        });
    }
    
    // Create evenly spaced time intervals (every 0.01 seconds = 100 samples per second for high precision)
    const timeInterval = 0.01; // 100 samples per second (10ms intervals)
    const timesToUse = [];
    for (let t = 0; t <= maxTime; t += timeInterval) {
        timesToUse.push({
            time: Math.min(t, maxTime) // Ensure we don't exceed maxTime
        });
    }
    
    // Always include the max time as the last row
    if (timesToUse.length === 0 || timesToUse[timesToUse.length - 1].time < maxTime) {
        timesToUse.push({ time: maxTime });
    }
    
    console.log('Using', timesToUse.length, 'evenly spaced time points (interval:', timeInterval, 's, max time:', maxTime.toFixed(3), 's)');
    console.log('Benchmark frames:', benchmarkPoseData.length, 'User frames:', userPoseData.length);
    
    // Track previous frame states to detect transitions
    let prevBenchWristAboveShoulder = null;
    let prevBenchElbowAboveShoulder = null;
    let prevBenchElbowBelowShoulder = null; // For shot end detection
    let prevUserWristAboveShoulder = null;
    let prevUserElbowAboveShoulder = null;
    let prevUserElbowBelowShoulder = null; // For shot end detection
    
    // Populate table for each time frame
    timesToUse.forEach((timeEntry, timeIdx) => {
        const currentTime = timeEntry.time;
        const row = document.createElement('tr');
        
        // Find closest benchmark frame by time (within reasonable tolerance)
        let benchPoseIdx = -1;
        let minBenchTimeDiff = Infinity;
        const maxTimeDiff = 0.05; // 50ms tolerance for matching frames
        benchmarkPoseData.forEach((frame, idx) => {
            if (frame && frame.time !== undefined && frame.time !== null) {
                const diff = Math.abs(frame.time - currentTime);
                if (diff < minBenchTimeDiff && diff <= maxTimeDiff) {
                    minBenchTimeDiff = diff;
                    benchPoseIdx = idx;
                }
            }
        });
        
        // Find closest user frame by time (within reasonable tolerance)
        let userPoseIdx = -1;
        let minUserTimeDiff = Infinity;
        userPoseData.forEach((frame, idx) => {
            if (frame && frame.time !== undefined && frame.time !== null) {
                const diff = Math.abs(frame.time - currentTime);
                if (diff < minUserTimeDiff && diff <= maxTimeDiff) {
                    minUserTimeDiff = diff;
                    userPoseIdx = idx;
                }
            }
        });
        
        // Get benchmark coordinates (null if no matching frame found)
        let benchWrist = null, benchElbow = null, benchShoulder = null;
        if (benchPoseIdx >= 0 && benchmarkPoseData[benchPoseIdx] && benchmarkPoseData[benchPoseIdx].landmarks) {
            const benchLandmarks = benchmarkPoseData[benchPoseIdx].landmarks;
            benchWrist = benchLandmarks[16]; // Right wrist
            benchElbow = benchLandmarks[14]; // Right elbow
            benchShoulder = benchLandmarks[12]; // Right shoulder
        }
        
        // Get user coordinates (null if no matching frame found)
        let userWrist = null, userElbow = null, userShoulder = null;
        if (userPoseIdx >= 0 && userPoseData[userPoseIdx] && userPoseData[userPoseIdx].landmarks) {
            const userLandmarks = userPoseData[userPoseIdx].landmarks;
            userWrist = userLandmarks[16]; // Right wrist
            userElbow = userLandmarks[14]; // Right elbow
            userShoulder = userLandmarks[12]; // Right shoulder
        }
        
        // Determine if this row should be highlighted based on transitions
        // Blue for Set Point: wrist Y transitions from > shoulder Y to <= shoulder Y
        // Yellow for Follow Through: elbow Y transitions from > shoulder Y to <= shoulder Y
        // Green for Shot End: elbow Y transitions from <= shoulder Y to > shoulder Y
        let isSetPoint = false;
        let isFollowThrough = false;
        let isShotEnd = false;
        let highlightReason = '';
        
        // Check benchmark Set Point transition: wrist Y changes from > shoulder Y (below) to <= shoulder Y (at/above)
        if (benchWrist && benchShoulder && Array.isArray(benchWrist) && Array.isArray(benchShoulder)) {
            const wristY = benchWrist[1];
            const shoulderY = benchShoulder[1];
            if (!isNaN(wristY) && !isNaN(shoulderY)) {
                const wristAboveShoulder = wristY <= shoulderY; // wrist is at or above shoulder
                // Initialize previous state on first frame if null
                if (prevBenchWristAboveShoulder === null) {
                    prevBenchWristAboveShoulder = wristAboveShoulder;
                } else {
                    // Transition: was below (wristY > shoulderY) to at/above (wristY <= shoulderY)
                    if (prevBenchWristAboveShoulder === false && wristAboveShoulder === true) {
                        isSetPoint = true;
                        transitions.benchSetPoint.push(currentTime);
                        highlightReason += `Benchmark Set Point (wrist crossed from below to at/above shoulder: wrist Y=${wristY.toFixed(2)}, shoulder Y=${shoulderY.toFixed(2)}). `;
                    }
                    prevBenchWristAboveShoulder = wristAboveShoulder;
                }
            }
        }
        
        // Check user Set Point transition: wrist Y changes from > shoulder Y (below) to <= shoulder Y (at/above)
        if (userWrist && userShoulder && Array.isArray(userWrist) && Array.isArray(userShoulder)) {
            const wristY = userWrist[1];
            const shoulderY = userShoulder[1];
            if (!isNaN(wristY) && !isNaN(shoulderY)) {
                const wristAboveShoulder = wristY <= shoulderY; // wrist is at or above shoulder
                // Initialize previous state on first frame if null
                if (prevUserWristAboveShoulder === null) {
                    prevUserWristAboveShoulder = wristAboveShoulder;
                } else {
                    // Transition: was below (wristY > shoulderY) to at/above (wristY <= shoulderY)
                    if (prevUserWristAboveShoulder === false && wristAboveShoulder === true) {
                        isSetPoint = true;
                        transitions.userSetPoint.push(currentTime);
                        highlightReason += `Your Shot Set Point (wrist crossed from below to at/above shoulder: wrist Y=${wristY.toFixed(2)}, shoulder Y=${shoulderY.toFixed(2)}). `;
                    }
                    prevUserWristAboveShoulder = wristAboveShoulder;
                }
            }
        }
        
        // Check benchmark Follow Through transition: elbow Y changes from > shoulder Y (below) to <= shoulder Y (at/above)
        if (benchElbow && benchShoulder && Array.isArray(benchElbow) && Array.isArray(benchShoulder)) {
            const elbowY = benchElbow[1];
            const shoulderY = benchShoulder[1];
            if (!isNaN(elbowY) && !isNaN(shoulderY)) {
                const elbowAboveShoulder = elbowY <= shoulderY; // elbow is at or above shoulder
                // Initialize previous state on first frame if null
                if (prevBenchElbowAboveShoulder === null) {
                    prevBenchElbowAboveShoulder = elbowAboveShoulder;
                } else {
                    // Transition: was below (elbowY > shoulderY) to at/above (elbowY <= shoulderY)
                    if (prevBenchElbowAboveShoulder === false && elbowAboveShoulder === true) {
                        isFollowThrough = true;
                        transitions.benchFollowThrough.push(currentTime);
                        highlightReason += `Benchmark Follow Through (elbow crossed from below to at/above shoulder: elbow Y=${elbowY.toFixed(2)}, shoulder Y=${shoulderY.toFixed(2)}). `;
                    }
                    prevBenchElbowAboveShoulder = elbowAboveShoulder;
                }
            }
        }
        
        // Check user Follow Through transition: elbow Y changes from > shoulder Y (below) to <= shoulder Y (at/above)
        if (userElbow && userShoulder && Array.isArray(userElbow) && Array.isArray(userShoulder)) {
            const elbowY = userElbow[1];
            const shoulderY = userShoulder[1];
            if (!isNaN(elbowY) && !isNaN(shoulderY)) {
                const elbowAboveShoulder = elbowY <= shoulderY; // elbow is at or above shoulder
                // Initialize previous state on first frame if null
                if (prevUserElbowAboveShoulder === null) {
                    prevUserElbowAboveShoulder = elbowAboveShoulder;
                } else {
                    // Transition: was below (elbowY > shoulderY) to at/above (elbowY <= shoulderY)
                    if (prevUserElbowAboveShoulder === false && elbowAboveShoulder === true) {
                        isFollowThrough = true;
                        transitions.userFollowThrough.push(currentTime);
                        highlightReason += `Your Shot Follow Through (elbow crossed from below to at/above shoulder: elbow Y=${elbowY.toFixed(2)}, shoulder Y=${shoulderY.toFixed(2)}). `;
                    }
                    prevUserElbowAboveShoulder = elbowAboveShoulder;
                }
            }
        }
        
        // Check benchmark Shot End transition: elbow Y changes from <= shoulder Y (at/above) to > shoulder Y (below)
        if (benchElbow && benchShoulder && Array.isArray(benchElbow) && Array.isArray(benchShoulder)) {
            const elbowY = benchElbow[1];
            const shoulderY = benchShoulder[1];
            if (!isNaN(elbowY) && !isNaN(shoulderY)) {
                const elbowBelowShoulder = elbowY > shoulderY; // elbow is below shoulder
                // Initialize previous state on first frame if null
                if (prevBenchElbowBelowShoulder === null) {
                    prevBenchElbowBelowShoulder = elbowBelowShoulder;
                } else {
                    // Transition: was at/above (elbowY <= shoulderY) to below (elbowY > shoulderY)
                    if (prevBenchElbowBelowShoulder === false && elbowBelowShoulder === true) {
                        isShotEnd = true;
                        transitions.benchShotEnd.push(currentTime);
                        highlightReason += `Benchmark Shot End (elbow crossed from at/above to below shoulder: elbow Y=${elbowY.toFixed(2)}, shoulder Y=${shoulderY.toFixed(2)}). `;
                    }
                    prevBenchElbowBelowShoulder = elbowBelowShoulder;
                }
            }
        }
        
        // Check user Shot End transition: elbow Y changes from <= shoulder Y (at/above) to > shoulder Y (below)
        if (userElbow && userShoulder && Array.isArray(userElbow) && Array.isArray(userShoulder)) {
            const elbowY = userElbow[1];
            const shoulderY = userShoulder[1];
            if (!isNaN(elbowY) && !isNaN(shoulderY)) {
                const elbowBelowShoulder = elbowY > shoulderY; // elbow is below shoulder
                // Initialize previous state on first frame if null
                if (prevUserElbowBelowShoulder === null) {
                    prevUserElbowBelowShoulder = elbowBelowShoulder;
                } else {
                    // Transition: was at/above (elbowY <= shoulderY) to below (elbowY > shoulderY)
                    if (prevUserElbowBelowShoulder === false && elbowBelowShoulder === true) {
                        isShotEnd = true;
                        transitions.userShotEnd.push(currentTime);
                        highlightReason += `Your Shot End (elbow crossed from at/above to below shoulder: elbow Y=${elbowY.toFixed(2)}, shoulder Y=${shoulderY.toFixed(2)}). `;
                    }
                    prevUserElbowBelowShoulder = elbowBelowShoulder;
                }
            }
        }
        
        // Apply color highlight: Blue for Set Point, Yellow for Follow Through, Green for Shot End
        // Priority: Set Point (blue) > Follow Through (yellow) > Shot End (green)
        if (isSetPoint) {
            row.style.backgroundColor = '#dbeafe'; // Light blue
            row.title = highlightReason.trim() || 'Set Point detected (wrist crossed shoulder level)';
            row.style.cursor = 'help';
        } else if (isFollowThrough) {
            row.style.backgroundColor = '#fef3c7'; // Light yellow
            row.title = highlightReason.trim() || 'Follow Through detected (elbow crossed shoulder level)';
            row.style.cursor = 'help';
        } else if (isShotEnd) {
            row.style.backgroundColor = '#d1fae5'; // Light green
            row.title = highlightReason.trim() || 'Shot End detected (elbow crossed below shoulder level)';
            row.style.cursor = 'help';
        }
        
        // Create cells
        const timeCell = document.createElement('td');
        timeCell.textContent = currentTime.toFixed(3);
        timeCell.style.padding = '8px';
        timeCell.style.border = '1px solid #e5e7eb';
        row.appendChild(timeCell);
        
        // Benchmark coordinates
        const benchWristCell = document.createElement('td');
        benchWristCell.textContent = formatCoordinates(benchWrist);
        benchWristCell.style.padding = '8px';
        benchWristCell.style.border = '1px solid #e5e7eb';
        benchWristCell.style.textAlign = 'center';
        benchWristCell.style.fontFamily = 'monospace';
        row.appendChild(benchWristCell);
        
        const benchElbowCell = document.createElement('td');
        benchElbowCell.textContent = formatCoordinates(benchElbow);
        benchElbowCell.style.padding = '8px';
        benchElbowCell.style.border = '1px solid #e5e7eb';
        benchElbowCell.style.textAlign = 'center';
        benchElbowCell.style.fontFamily = 'monospace';
        row.appendChild(benchElbowCell);
        
        const benchShoulderCell = document.createElement('td');
        benchShoulderCell.textContent = formatCoordinates(benchShoulder);
        benchShoulderCell.style.padding = '8px';
        benchShoulderCell.style.border = '1px solid #e5e7eb';
        benchShoulderCell.style.textAlign = 'center';
        benchShoulderCell.style.fontFamily = 'monospace';
        row.appendChild(benchShoulderCell);
        
        // User coordinates
        const userWristCell = document.createElement('td');
        userWristCell.textContent = formatCoordinates(userWrist);
        userWristCell.style.padding = '8px';
        userWristCell.style.border = '1px solid #e5e7eb';
        userWristCell.style.textAlign = 'center';
        userWristCell.style.fontFamily = 'monospace';
        row.appendChild(userWristCell);
        
        const userElbowCell = document.createElement('td');
        userElbowCell.textContent = formatCoordinates(userElbow);
        userElbowCell.style.padding = '8px';
        userElbowCell.style.border = '1px solid #e5e7eb';
        userElbowCell.style.textAlign = 'center';
        userElbowCell.style.fontFamily = 'monospace';
        row.appendChild(userElbowCell);
        
        const userShoulderCell = document.createElement('td');
        userShoulderCell.textContent = formatCoordinates(userShoulder);
        userShoulderCell.style.padding = '8px';
        userShoulderCell.style.border = '1px solid #e5e7eb';
        userShoulderCell.style.textAlign = 'center';
        userShoulderCell.style.fontFamily = 'monospace';
        row.appendChild(userShoulderCell);
        
        tableBody.appendChild(row);
    });
    
    console.log('Coordinate table populated with', timesToUse.length, 'rows');
    console.log('Detected transitions:', transitions);
    
    return transitions;
}

function displayResults(data) {
    console.log('displayResults called with data:', data);
    const loadingEl = document.getElementById('loading');
    const resultsEl = document.getElementById('results');
    
    if (loadingEl) loadingEl.style.display = 'none';
    if (resultsEl) {
        resultsEl.style.display = 'block';
        console.log('Results section displayed');
    } else {
        console.error('results element not found!');
    }
    
    // Calculate average score from actual data
    const avgCloseness = data.userCloseness.reduce((a, b) => a + b, 0) / data.userCloseness.length;
    
    // Add player name to title if applicable
    const playerNames = {
        'curry': 'Stephen Curry',
        'lebron': 'LeBron James',
        'jordan': 'Michael Jordan',
        'durant': 'Kevin Durant',
        'clark': 'Caitlin Clark'
    };
    
    let title = `Overall Score: ${avgCloseness.toFixed(1)}%`;
    if (data.playerName && data.playerName !== 'custom') {
        title += ` (vs ${playerNames[data.playerName]})`;
    }
    document.getElementById('overallScore').textContent = title;
    
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    // Create gradient for user's shot line
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(255, 107, 122, 0.3)');
    gradient.addColorStop(1, 'rgba(255, 107, 122, 0.05)');

    // Stage colors for dots
    const stageColors = {
        shot_start: '#3b82f6', // Blue
        set_point: '#fbbf24', // Yellow
        follow_through: '#10b981' // Green
    };
    const stageLabels = {
        shot_start: 'Shot Start',
        set_point: 'Set Point',
        follow_through: 'Follow Through'
    };
    
    // Prepare data arrays with point markers
    const benchmarkData = data.userTimes.map(() => 100);
    const userData = [...data.userCloseness];
    
    // Arrays to store point styles for each data point
    const benchmarkPointRadius = new Array(data.userTimes.length).fill(0);
    const benchmarkPointBackgroundColor = new Array(data.userTimes.length).fill('transparent');
    const benchmarkPointBorderColor = new Array(data.userTimes.length).fill('transparent');
    
    const userPointRadius = new Array(data.userTimes.length).fill(0);
    const userPointBackgroundColor = new Array(data.userTimes.length).fill('transparent');
    const userPointBorderColor = new Array(data.userTimes.length).fill('transparent');
    
    // Populate coordinate table first to get detected transitions
    const transitions = populateCoordinateTable(data);
    
    // Add transition dots to graph arrays based on coordinate table transitions
    if (transitions) {
        // Add benchmark Set Point dots (blue)
        transitions.benchSetPoint.forEach(time => {
            let closestIdx = 0;
            let minDiff = Math.abs(data.userTimes[0] - time);
            for (let i = 1; i < data.userTimes.length; i++) {
                const diff = Math.abs(data.userTimes[i] - time);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = i;
                }
            }
            if (closestIdx < benchmarkPointRadius.length) {
                benchmarkPointRadius[closestIdx] = 8;
                benchmarkPointBackgroundColor[closestIdx] = '#3b82f6'; // Blue
                benchmarkPointBorderColor[closestIdx] = '#3b82f6';
            }
        });
        
        // Add benchmark Follow Through dots (yellow)
        transitions.benchFollowThrough.forEach(time => {
            let closestIdx = 0;
            let minDiff = Math.abs(data.userTimes[0] - time);
            for (let i = 1; i < data.userTimes.length; i++) {
                const diff = Math.abs(data.userTimes[i] - time);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = i;
                }
            }
            if (closestIdx < benchmarkPointRadius.length) {
                benchmarkPointRadius[closestIdx] = 8;
                benchmarkPointBackgroundColor[closestIdx] = '#fbbf24'; // Yellow
                benchmarkPointBorderColor[closestIdx] = '#fbbf24';
            }
        });
        
        // Add user Set Point dots (blue)
        transitions.userSetPoint.forEach(time => {
            let closestIdx = 0;
            let minDiff = Math.abs(data.userTimes[0] - time);
            for (let i = 1; i < data.userTimes.length; i++) {
                const diff = Math.abs(data.userTimes[i] - time);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = i;
                }
            }
            if (closestIdx < userPointRadius.length) {
                userPointRadius[closestIdx] = 8;
                userPointBackgroundColor[closestIdx] = '#3b82f6'; // Blue
                userPointBorderColor[closestIdx] = '#3b82f6';
            }
        });
        
        // Add user Follow Through dots (yellow)
        transitions.userFollowThrough.forEach(time => {
            let closestIdx = 0;
            let minDiff = Math.abs(data.userTimes[0] - time);
            for (let i = 1; i < data.userTimes.length; i++) {
                const diff = Math.abs(data.userTimes[i] - time);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = i;
                }
            }
            if (closestIdx < userPointRadius.length) {
                userPointRadius[closestIdx] = 8;
                userPointBackgroundColor[closestIdx] = '#fbbf24'; // Yellow
                userPointBorderColor[closestIdx] = '#fbbf24';
            }
        });
        
        // Add benchmark Shot End dots (green)
        transitions.benchShotEnd.forEach(time => {
            let closestIdx = 0;
            let minDiff = Math.abs(data.userTimes[0] - time);
            for (let i = 1; i < data.userTimes.length; i++) {
                const diff = Math.abs(data.userTimes[i] - time);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = i;
                }
            }
            if (closestIdx < benchmarkPointRadius.length) {
                benchmarkPointRadius[closestIdx] = 8;
                benchmarkPointBackgroundColor[closestIdx] = '#10b981'; // Green
                benchmarkPointBorderColor[closestIdx] = '#10b981';
            }
        });
        
        // Add user Shot End dots (green)
        transitions.userShotEnd.forEach(time => {
            let closestIdx = 0;
            let minDiff = Math.abs(data.userTimes[0] - time);
            for (let i = 1; i < data.userTimes.length; i++) {
                const diff = Math.abs(data.userTimes[i] - time);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = i;
                }
            }
            if (closestIdx < userPointRadius.length) {
                userPointRadius[closestIdx] = 8;
                userPointBackgroundColor[closestIdx] = '#10b981'; // Green
                userPointBorderColor[closestIdx] = '#10b981';
            }
        });
        
        console.log('Added transition dots to graph:', {
            benchSetPoint: transitions.benchSetPoint.length,
            benchFollowThrough: transitions.benchFollowThrough.length,
            benchShotEnd: transitions.benchShotEnd.length,
            userSetPoint: transitions.userSetPoint.length,
            userFollowThrough: transitions.userFollowThrough.length,
            userShotEnd: transitions.userShotEnd.length
        });
    }
    
    // Only use transition dots from highlighted rows (no old stage markers)
    console.log('Benchmark point radius array:', benchmarkPointRadius.filter(r => r > 0).length, 'non-zero values');
    console.log('User point radius array:', userPointRadius.filter(r => r > 0).length, 'non-zero values');
    
    comparisonChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.userTimes.map(t => t.toFixed(2)),
            datasets: [{
                label: 'Benchmark (100%)',
                data: benchmarkData,
                borderColor: 'rgba(147, 112, 219, 0.8)',
                backgroundColor: 'rgba(147, 112, 219, 0.1)',
                borderDash: [8, 4],
                borderWidth: 2,
                pointRadius: (ctx) => {
                    return benchmarkPointRadius[ctx.dataIndex] || 0;
                },
                pointBackgroundColor: (ctx) => {
                    return benchmarkPointBackgroundColor[ctx.dataIndex] || 'transparent';
                },
                pointBorderColor: (ctx) => {
                    return benchmarkPointBorderColor[ctx.dataIndex] || 'transparent';
                },
                pointBorderWidth: 2,
                pointHoverRadius: 10,
                tension: 0.4
            }, {
                label: 'Your Shot',
                data: userData,
                borderColor: 'rgb(255, 107, 122)',
                backgroundColor: gradient,
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: (ctx) => {
                    return userPointRadius[ctx.dataIndex] || 0;
                },
                pointBackgroundColor: (ctx) => {
                    return userPointBackgroundColor[ctx.dataIndex] || 'transparent';
                },
                pointBorderColor: (ctx) => {
                    return userPointBorderColor[ctx.dataIndex] || 'transparent';
                },
                pointBorderWidth: 2,
                pointHoverRadius: 10,
                pointHoverBackgroundColor: 'rgb(255, 107, 122)',
                pointHoverBorderColor: 'white',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Shot Form Over Time',
                    font: {
                        size: 18,
                        weight: '600',
                        family: "'Inter', sans-serif"
                    },
                    color: '#1f2937',
                    padding: {
                        top: 10,
                        bottom: 20
                    }
                },
                legend: {
                    display: true,
                    position: 'right',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 13,
                            family: "'Inter', sans-serif"
                        },
                        generateLabels: function(chart) {
                            // Custom legend with stage markers (no shot_start since they start together)
                            return [
                                {
                                    text: 'Benchmark (100%)',
                                    fillStyle: 'rgba(147, 112, 219, 0.8)',
                                    strokeStyle: 'rgba(147, 112, 219, 0.8)',
                                    lineDash: [8, 4],
                                    pointStyle: 'line'
                                },
                                {
                                    text: 'Your Shot',
                                    fillStyle: 'rgb(255, 107, 122)',
                                    strokeStyle: 'rgb(255, 107, 122)',
                                    pointStyle: 'line'
                                },
                                {
                                    text: ' ', // Spacer
                                    fillStyle: 'transparent',
                                    strokeStyle: 'transparent'
                                },
                                {
                                    text: 'Set Point',
                                    fillStyle: '#3b82f6', // Blue
                                    strokeStyle: '#3b82f6',
                                    pointStyle: 'circle'
                                },
                                {
                                    text: 'Follow Through',
                                    fillStyle: '#fbbf24', // Yellow
                                    strokeStyle: '#fbbf24',
                                    pointStyle: 'circle'
                                },
                                {
                                    text: 'Shot End',
                                    fillStyle: '#10b981', // Green
                                    strokeStyle: '#10b981',
                                    pointStyle: 'circle'
                                }
                            ];
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: {
                        size: 14,
                        weight: '600'
                    },
                    bodyFont: {
                        size: 13
                    },
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 110,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.06)',
                        drawBorder: false
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 12,
                            family: "'Inter', sans-serif"
                        },
                        color: '#6b7280'
                    },
                    title: {
                        display: true,
                        text: 'Similarity to Benchmark',
                        font: {
                            size: 13,
                            weight: '600',
                            family: "'Inter', sans-serif"
                        },
                        color: '#4b5563',
                        padding: {
                            top: 10,
                            bottom: 0
                        }
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.06)',
                        drawBorder: false
                    },
                    ticks: {
                        maxRotation: 0,
                        autoSkipPadding: 20,
                        font: {
                            size: 12,
                            family: "'Inter', sans-serif"
                        },
                        color: '#6b7280'
                    },
                    title: {
                        display: true,
                        text: 'Time (seconds)',
                        font: {
                            size: 13,
                            weight: '600',
                            family: "'Inter', sans-serif"
                        },
                        color: '#4b5563',
                        padding: {
                            top: 10,
                            bottom: 0
                        }
                    }
                }
            }
        }
    });
    
    // Display stage-based angle analysis tabs
    displayStageAngleTabs(data);
    
    // Generate and display detailed feedback
    try {
        console.log('Generating feedback for player:', data.playerName, 'Data:', data);
        const detailedFeedback = generatePlayerSpecificFeedback(data);
        console.log('Generated feedback:', detailedFeedback);
        displayDetailedFeedback(detailedFeedback, data.playerName);
    } catch (error) {
        console.error('Error generating/displaying feedback:', error);
        // Fallback: show basic feedback
        const detailedFeedbackSection = document.getElementById('detailedFeedback');
        if (detailedFeedbackSection) {
            detailedFeedbackSection.style.display = 'block';
            const summaryEl = document.getElementById('shotSummary');
            if (summaryEl) {
                const avgCloseness = data.userCloseness.reduce((a, b) => a + b, 0) / data.userCloseness.length;
                summaryEl.textContent = `Your shot analysis shows ${avgCloseness.toFixed(1)}% similarity to the benchmark.`;
            }
        }
    }
    
    // Hide old feedback section
    const oldFeedbackSection = document.getElementById('oldFeedbackSection');
    if (oldFeedbackSection) {
        oldFeedbackSection.style.display = 'none';
    }
}

// Display tabs showing angles for each shooting stage
function displayStageAngleTabs(data) {
    let stageAngleContainer = document.getElementById('stageAngleTabs');
    if (!stageAngleContainer) {
        // Create the container if it doesn't exist
        const resultsEl = document.getElementById('results');
        if (resultsEl) {
            const container = document.createElement('div');
            container.id = 'stageAngleTabs';
            container.style.cssText = 'margin-top: 40px;';
            resultsEl.appendChild(container);
            stageAngleContainer = container;
        } else {
            return;
        }
    }
    
    const stages = [
        { key: 'shot_start', label: 'Shot Start', color: '#10b981' },
        { key: 'set_point', label: 'Set Point', color: '#3b82f6' },
        { key: 'follow_through', label: 'Follow Through', color: '#f59e0b' }
    ];
    
    // Create tabs HTML
    let tabsHTML = '<div class="stage-tabs-container" style="margin-top: 40px; margin-bottom: 30px;">';
    tabsHTML += '<h4 style="margin-bottom: 20px; font-size: 1.4em; font-weight: 700; color: #1a202c;">Angle Analysis by Stage</h4>';
    tabsHTML += '<div class="stage-tabs" style="display: flex; gap: 10px; border-bottom: 2px solid #e2e8f0; margin-bottom: 20px;">';
    
    stages.forEach((stage, idx) => {
        tabsHTML += `<button class="stage-tab-btn" data-stage="${stage.key}" style="
            padding: 12px 24px;
            border: none;
            background: ${idx === 0 ? stage.color : 'transparent'};
            color: ${idx === 0 ? 'white' : '#4a5568'};
            font-weight: ${idx === 0 ? '600' : '500'};
            font-size: 14px;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            transition: all 0.3s ease;
            border-bottom: 3px solid ${idx === 0 ? stage.color : 'transparent'};
        ">${stage.label}</button>`;
    });
    
    tabsHTML += '</div>';
    
    // Create tab content
    tabsHTML += '<div class="stage-tab-content" style="min-height: 200px;">';
    stages.forEach((stage, idx) => {
        const userMarker = data.userStageMarkers && data.userStageMarkers[stage.key];
        const benchMarker = data.benchStageMarkers && data.benchStageMarkers[stage.key];
        
        tabsHTML += `<div class="stage-tab-panel" data-stage="${stage.key}" style="display: ${idx === 0 ? 'block' : 'none'}; padding: 20px; background: #f8fafc; border-radius: 8px;">`;
        
        if (userMarker && benchMarker) {
            // Calculate differences
            const elbowDiff = userMarker.elbow_angle - benchMarker.elbow_angle;
            const wristDiff = userMarker.wrist_angle - benchMarker.wrist_angle;
            const armDiff = userMarker.arm_angle - benchMarker.arm_angle;
            
            tabsHTML += `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                    <div class="angle-comparison-card" style="background: white; padding: 20px; border-radius: 12px; border-left: 4px solid ${stage.color};">
                        <h5 style="margin: 0 0 15px 0; color: #1a202c; font-size: 1.1em; font-weight: 600;">Elbow Angle</h5>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #718096;">Your:</span>
                            <span style="font-weight: 600; color: ${elbowDiff > 10 || elbowDiff < -10 ? '#ef4444' : '#10b981'};">${userMarker.elbow_angle ? userMarker.elbow_angle.toFixed(1) : 'N/A'}°</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #718096;">Benchmark:</span>
                            <span style="font-weight: 600; color: #4a5568;">${benchMarker.elbow_angle ? benchMarker.elbow_angle.toFixed(1) : 'N/A'}°</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding-top: 10px; border-top: 1px solid #e2e8f0;">
                            <span style="color: #718096;">Difference:</span>
                            <span style="font-weight: 600; color: ${Math.abs(elbowDiff) > 10 ? '#ef4444' : '#10b981'};">${elbowDiff > 0 ? '+' : ''}${elbowDiff.toFixed(1)}°</span>
                        </div>
                    </div>
                    
                    <div class="angle-comparison-card" style="background: white; padding: 20px; border-radius: 12px; border-left: 4px solid ${stage.color};">
                        <h5 style="margin: 0 0 15px 0; color: #1a202c; font-size: 1.1em; font-weight: 600;">Wrist Angle</h5>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #718096;">Your:</span>
                            <span style="font-weight: 600; color: ${wristDiff > 10 || wristDiff < -10 ? '#ef4444' : '#10b981'};">${userMarker.wrist_angle ? userMarker.wrist_angle.toFixed(1) : 'N/A'}°</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #718096;">Benchmark:</span>
                            <span style="font-weight: 600; color: #4a5568;">${benchMarker.wrist_angle ? benchMarker.wrist_angle.toFixed(1) : 'N/A'}°</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding-top: 10px; border-top: 1px solid #e2e8f0;">
                            <span style="color: #718096;">Difference:</span>
                            <span style="font-weight: 600; color: ${Math.abs(wristDiff) > 10 ? '#ef4444' : '#10b981'};">${wristDiff > 0 ? '+' : ''}${wristDiff.toFixed(1)}°</span>
                        </div>
                    </div>
                    
                    <div class="angle-comparison-card" style="background: white; padding: 20px; border-radius: 12px; border-left: 4px solid ${stage.color};">
                        <h5 style="margin: 0 0 15px 0; color: #1a202c; font-size: 1.1em; font-weight: 600;">Arm Angle</h5>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #718096;">Your:</span>
                            <span style="font-weight: 600; color: ${armDiff > 10 || armDiff < -10 ? '#ef4444' : '#10b981'};">${userMarker.arm_angle ? userMarker.arm_angle.toFixed(1) : 'N/A'}°</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #718096;">Benchmark:</span>
                            <span style="font-weight: 600; color: #4a5568;">${benchMarker.arm_angle ? benchMarker.arm_angle.toFixed(1) : 'N/A'}°</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding-top: 10px; border-top: 1px solid #e2e8f0;">
                            <span style="color: #718096;">Difference:</span>
                            <span style="font-weight: 600; color: ${Math.abs(armDiff) > 10 ? '#ef4444' : '#10b981'};">${armDiff > 0 ? '+' : ''}${armDiff.toFixed(1)}°</span>
                        </div>
                    </div>
                </div>
            `;
        } else {
            tabsHTML += `<p style="color: #718096; text-align: center; padding: 40px;">No data available for ${stage.label} stage.</p>`;
        }
        
        tabsHTML += '</div>';
    });
    
    tabsHTML += '</div></div>';
    
    stageAngleContainer.innerHTML = tabsHTML;
    
    // Add tab switching functionality
    const tabButtons = stageAngleContainer.querySelectorAll('.stage-tab-btn');
    const tabPanels = stageAngleContainer.querySelectorAll('.stage-tab-panel');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const stage = btn.dataset.stage;
            
            // Update button styles
            tabButtons.forEach(b => {
                const stageData = stages.find(s => s.key === b.dataset.stage);
                b.style.background = 'transparent';
                b.style.color = '#4a5568';
                b.style.fontWeight = '500';
                b.style.borderBottom = '3px solid transparent';
            });
            
            const stageData = stages.find(s => s.key === stage);
            btn.style.background = stageData.color;
            btn.style.color = 'white';
            btn.style.fontWeight = '600';
            btn.style.borderBottom = `3px solid ${stageData.color}`;
            
            // Show/hide panels
            tabPanels.forEach(panel => {
                panel.style.display = panel.dataset.stage === stage ? 'block' : 'none';
            });
        });
    });
}

function generatePlayerSpecificFeedback(data) {
    const player = data.playerName;
    if (!player || player === 'custom') {
        return generateGenericFeedback(data);
    }
    
    // Calculate average metrics from recorded data
    const avgCloseness = data.userCloseness.reduce((a, b) => a + b, 0) / data.userCloseness.length;
    let avgElbowAngle = data.avgElbowAngle || 0;
    let avgWristAngle = data.avgWristAngle || 0;
    let avgArmAngle = data.avgArmAngle || 0;
    
    const playerFeedback = {
        'curry': {
            name: 'Stephen Curry',
            niche: 'One-Motion Shot',
            idealElbow: 150,
            idealWrist: 90,
            idealArm: 50,
            strengths: [],
            weaknesses: [],
            metrics: [],
            summary: ''
        },
        'lebron': {
            name: 'LeBron James',
            niche: 'Power & Elevation',
            idealElbow: 140,
            idealWrist: 95,
            idealArm: 55,
            strengths: [],
            weaknesses: [],
            metrics: [],
            summary: ''
        },
        'jordan': {
            name: 'Michael Jordan',
            niche: 'Classic Form & Hang Time',
            idealElbow: 145,
            idealWrist: 92,
            idealArm: 52,
            strengths: [],
            weaknesses: [],
            metrics: [],
            summary: ''
        },
        'durant': {
            name: 'Kevin Durant',
            niche: 'High Release Point',
            idealElbow: 155,
            idealWrist: 88,
            idealArm: 48,
            strengths: [],
            weaknesses: [],
            metrics: [],
            summary: ''
        },
        'clark': {
            name: 'Caitlin Clark',
            niche: 'Quick Release & Range',
            idealElbow: 148,
            idealWrist: 91,
            idealArm: 51,
            strengths: [],
            weaknesses: [],
            metrics: [],
            summary: ''
        }
    };
    
    const feedback = playerFeedback[player];
    if (!feedback) return generateGenericFeedback(data);
    
    // Calculate differences from ideal
    const elbowDiff = Math.abs(avgElbowAngle - feedback.idealElbow);
    const wristDiff = Math.abs(avgWristAngle - feedback.idealWrist);
    const armDiff = Math.abs(avgArmAngle - feedback.idealArm);
    
    // Generate strengths
    if (elbowDiff < 15) {
        feedback.strengths.push({
            title: 'Elbow Extension',
            value: `${avgElbowAngle.toFixed(1)}°`,
            ideal: `${feedback.idealElbow}°`,
            score: Math.max(0, 100 - (elbowDiff * 2))
        });
    }
    if (wristDiff < 10) {
        feedback.strengths.push({
            title: 'Wrist Snap',
            value: `${avgWristAngle.toFixed(1)}°`,
            ideal: `${feedback.idealWrist}°`,
            score: Math.max(0, 100 - (wristDiff * 3))
        });
    }
    if (armDiff < 8) {
        feedback.strengths.push({
            title: 'Arm Angle',
            value: `${avgArmAngle.toFixed(1)}°`,
            ideal: `${feedback.idealArm}°`,
            score: Math.max(0, 100 - (armDiff * 4))
        });
    }
    if (avgCloseness > 75) {
        feedback.strengths.push({
            title: 'Overall Form',
            value: `${avgCloseness.toFixed(1)}%`,
            ideal: '100%',
            score: avgCloseness
        });
    }
    
    // Generate weaknesses
    if (elbowDiff >= 15) {
        feedback.weaknesses.push({
            title: 'Elbow Extension',
            value: `${avgElbowAngle.toFixed(1)}°`,
            ideal: `${feedback.idealElbow}°`,
            score: Math.max(0, 100 - (elbowDiff * 2)),
            tip: player === 'curry' ? 'Focus on a smooth, continuous motion from your set point to release. Curry\'s one-motion shot requires full elbow extension.' : 'Work on fully extending your elbow at the point of release.'
        });
    }
    if (wristDiff >= 10) {
        feedback.weaknesses.push({
            title: 'Wrist Snap',
            value: `${avgWristAngle.toFixed(1)}°`,
            ideal: `${feedback.idealWrist}°`,
            score: Math.max(0, 100 - (wristDiff * 3)),
            tip: player === 'curry' ? 'The wrist snap is crucial for Curry\'s one-motion shot. Practice a quick, decisive flick at the end of your shooting motion.' : 'Improve your wrist snap timing and angle for better ball control.'
        });
    }
    if (armDiff >= 8) {
        feedback.weaknesses.push({
            title: 'Arm Angle',
            value: `${avgArmAngle.toFixed(1)}°`,
            ideal: `${feedback.idealArm}°`,
            score: Math.max(0, 100 - (armDiff * 4)),
            tip: player === 'durant' ? 'Durant\'s high release comes from optimal arm angle. Keep your shooting arm at the right angle for maximum elevation.' : 'Adjust your arm angle to match the ideal shooting form.'
        });
    }
    if (avgCloseness < 75) {
        feedback.weaknesses.push({
            title: 'Overall Consistency',
            value: `${avgCloseness.toFixed(1)}%`,
            ideal: '100%',
            score: avgCloseness,
            tip: 'Focus on maintaining consistent form throughout your shooting motion.'
        });
    }
    
    // Player-specific niche comparison
    let nicheScore = 0;
    let nicheFeedback = '';
    
    if (player === 'curry') {
        // One-motion shot: smooth transition, quick release
        const releaseSpeed = data.userTimes.length > 0 ? data.userTimes[data.userTimes.length - 1] - data.userTimes[0] : 0;
        const smoothness = avgCloseness;
        nicheScore = (smoothness * 0.7) + (Math.min(100, 100 - releaseSpeed * 10) * 0.3);
        nicheFeedback = `Your one-motion shot similarity: ${nicheScore.toFixed(1)}%. ${nicheScore > 70 ? 'Great job emulating Curry\'s signature smooth, continuous motion!' : 'Work on creating a more fluid, uninterrupted shooting motion from start to finish.'}`;
    } else if (player === 'lebron') {
        // Power & elevation: strong extension, high release
        nicheScore = (avgElbowAngle / 180 * 100 * 0.6) + (avgCloseness * 0.4);
        nicheFeedback = `Your power shot similarity: ${nicheScore.toFixed(1)}%. ${nicheScore > 70 ? 'Excellent power and elevation in your shot!' : 'Focus on generating more power through your legs and extending fully for maximum elevation.'}`;
    } else if (player === 'jordan') {
        // Classic form: textbook mechanics
        nicheScore = avgCloseness;
        nicheFeedback = `Your classic form similarity: ${nicheScore.toFixed(1)}%. ${nicheScore > 75 ? 'You\'re displaying textbook shooting mechanics like MJ!' : 'Focus on the fundamentals: balance, follow-through, and consistent form.'}`;
    } else if (player === 'durant') {
        // High release point: arm angle and extension combined
        nicheScore = ((180 - Math.abs(avgArmAngle - 48)) / 180 * 100 * 0.5) + (avgCloseness * 0.5);
        nicheFeedback = `Your high release similarity: ${nicheScore.toFixed(1)}%. ${nicheScore > 70 ? 'Great job getting your release point high like Durant!' : 'Work on elevating your release point by adjusting your arm angle and extension.'}`;
    } else if (player === 'clark') {
        // Quick release & range: fast motion, good extension
        const releaseSpeed = data.userTimes.length > 0 ? data.userTimes[data.userTimes.length - 1] - data.userTimes[0] : 0;
        nicheScore = (Math.min(100, 100 - releaseSpeed * 15) * 0.5) + (avgCloseness * 0.5);
        nicheFeedback = `Your quick release similarity: ${nicheScore.toFixed(1)}%. ${nicheScore > 70 ? 'Excellent quick release and range!' : 'Practice a faster, more efficient shooting motion while maintaining accuracy.'}`;
    }
    
    // Generate summary
    feedback.summary = `Compared to ${feedback.name}'s ${feedback.niche}, your shot shows ${avgCloseness > 75 ? 'strong' : avgCloseness > 60 ? 'moderate' : 'room for improvement in'} similarity. ${nicheFeedback}`;
    
    // Add all metrics
    feedback.metrics = [
        { label: 'Elbow Extension', value: `${avgElbowAngle.toFixed(1)}°`, ideal: `${feedback.idealElbow}°`, score: Math.max(0, 100 - elbowDiff * 2) },
        { label: 'Wrist Snap', value: `${avgWristAngle.toFixed(1)}°`, ideal: `${feedback.idealWrist}°`, score: Math.max(0, 100 - wristDiff * 3) },
        { label: 'Arm Angle', value: `${avgArmAngle.toFixed(1)}°`, ideal: `${feedback.idealArm}°`, score: Math.max(0, 100 - armDiff * 4) },
        { label: 'Overall Score', value: `${avgCloseness.toFixed(1)}%`, ideal: '100%', score: avgCloseness },
        { label: `${feedback.niche} Similarity`, value: `${nicheScore.toFixed(1)}%`, ideal: '100%', score: nicheScore }
    ];
    
    return feedback;
}

function generateGenericFeedback(data) {
    const avgCloseness = data.userCloseness.reduce((a, b) => a + b, 0) / data.userCloseness.length;
    return {
        name: 'Benchmark',
        niche: 'Standard Form',
        strengths: avgCloseness > 70 ? [{ title: 'Overall Form', value: `${avgCloseness.toFixed(1)}%`, ideal: '100%', score: avgCloseness }] : [],
        weaknesses: avgCloseness < 70 ? [{ title: 'Overall Consistency', value: `${avgCloseness.toFixed(1)}%`, ideal: '100%', score: avgCloseness, tip: 'Focus on maintaining consistent form throughout your shooting motion.' }] : [],
        metrics: [{ label: 'Overall Score', value: `${avgCloseness.toFixed(1)}%`, ideal: '100%', score: avgCloseness }],
        summary: `Your shot analysis shows ${avgCloseness > 75 ? 'strong' : avgCloseness > 60 ? 'moderate' : 'room for improvement in'} similarity to the benchmark. Keep practicing to improve your consistency!`
    };
}

function displayDetailedFeedback(feedback, playerName) {
    console.log('displayDetailedFeedback called with:', { feedback, playerName });
    
    // Try multiple times to find the element (in case DOM isn't ready)
    let detailedFeedbackSection = document.getElementById('detailedFeedback');
    
    if (!detailedFeedbackSection) {
        console.error('detailedFeedback element not found, retrying...');
        setTimeout(() => {
            detailedFeedbackSection = document.getElementById('detailedFeedback');
            if (detailedFeedbackSection) {
                console.log('Found detailedFeedback element on retry');
                detailedFeedbackSection.style.display = 'block';
                populateFeedbackContent(feedback, playerName);
    } else {
                console.error('detailedFeedback element still not found after retry');
            }
        }, 200);
        return;
    }
    
    console.log('Found detailedFeedback element, displaying...');
    detailedFeedbackSection.style.display = 'block';
    detailedFeedbackSection.style.visibility = 'visible';
    
    // Force a reflow to ensure it's visible
    detailedFeedbackSection.offsetHeight;
    
    populateFeedbackContent(feedback, playerName);
}

function populateFeedbackContent(feedback, playerName) {
    console.log('populateFeedbackContent called with:', { feedback, playerName });
    
    if (!feedback) {
        console.error('No feedback data provided');
        return;
    }
    
    // Set player comparison title
    const titleEl = document.getElementById('playerComparisonTitle');
    if (titleEl) {
        if (playerName && playerName !== 'custom' && feedback.name) {
            titleEl.textContent = `Comparing your shot to ${feedback.name}'s ${feedback.niche}`;
        } else {
            titleEl.textContent = 'Detailed shot analysis';
        }
        console.log('Set title to:', titleEl.textContent);
    } else {
        console.error('playerComparisonTitle element not found');
    }
    
    // Display summary
    const summaryEl = document.getElementById('shotSummary');
    if (summaryEl) {
        if (feedback.summary) {
            summaryEl.textContent = feedback.summary;
            console.log('Set summary:', feedback.summary);
        } else {
            summaryEl.textContent = 'Shot analysis complete.';
            console.warn('No summary in feedback, using default');
        }
    } else {
        console.error('shotSummary element not found');
    }
    
    // Display strengths
    const strengthsList = document.getElementById('strengthsList');
    if (strengthsList) {
        strengthsList.innerHTML = '';
        if (feedback.strengths && feedback.strengths.length > 0) {
            console.log('Displaying', feedback.strengths.length, 'strengths');
            feedback.strengths.forEach(strength => {
                const item = document.createElement('div');
                item.className = 'feedback-item strength-item';
                item.innerHTML = `
                    <div class="feedback-item-header">
                        <span class="feedback-title">${strength.title}</span>
                        <span class="feedback-score">${strength.score.toFixed(0)}%</span>
                    </div>
                    <div class="feedback-values">
                        <span class="feedback-value">${strength.value}</span>
                        <span class="feedback-ideal">${strength.ideal}</span>
                    </div>
                `;
                strengthsList.appendChild(item);
            });
        } else {
            console.log('No strengths, showing placeholder');
            strengthsList.innerHTML = '<p class="no-feedback">Keep practicing to develop your strengths!</p>';
        }
    } else {
        console.error('strengthsList element not found');
    }
    
    // Display weaknesses
    const weaknessesList = document.getElementById('weaknessesList');
    if (weaknessesList) {
        weaknessesList.innerHTML = '';
        if (feedback.weaknesses && feedback.weaknesses.length > 0) {
            console.log('Displaying', feedback.weaknesses.length, 'weaknesses');
            feedback.weaknesses.forEach(weakness => {
                const item = document.createElement('div');
                item.className = 'feedback-item weakness-item';
                item.innerHTML = `
                    <div class="feedback-item-header">
                        <span class="feedback-title">${weakness.title}</span>
                        <span class="feedback-score">${weakness.score.toFixed(0)}%</span>
                    </div>
                    <div class="feedback-values">
                        <span class="feedback-value">${weakness.value}</span>
                        <span class="feedback-ideal">${weakness.ideal}</span>
                    </div>
                    ${weakness.tip ? `<p class="feedback-tip">💡 ${weakness.tip}</p>` : ''}
                `;
                weaknessesList.appendChild(item);
            });
        } else {
            console.log('No weaknesses, showing placeholder');
            weaknessesList.innerHTML = '<p class="no-feedback">Excellent! No major areas need improvement.</p>';
        }
    } else {
        console.error('weaknessesList element not found');
    }
    
    // Display metrics
    const metricsList = document.getElementById('metricsList');
    if (metricsList) {
        if (feedback.metrics && feedback.metrics.length > 0) {
            console.log('Displaying', feedback.metrics.length, 'metrics');
            metricsList.innerHTML = '';
            feedback.metrics.forEach(metric => {
                const item = document.createElement('div');
                item.className = 'metric-item';
                item.innerHTML = `
                    <div class="metric-label">${metric.label}</div>
                    <div class="metric-value">${metric.value}</div>
                    <div class="metric-ideal">Ideal: ${metric.ideal}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="width: ${Math.max(0, Math.min(100, metric.score))}%"></div>
                    </div>
                    <div class="metric-score">${metric.score.toFixed(0)}%</div>
                `;
                metricsList.appendChild(item);
            });
        } else {
            console.warn('No metrics in feedback');
            metricsList.innerHTML = '<p class="no-feedback">Metrics not available.</p>';
        }
    } else {
        console.error('metricsList element not found');
    }
    
    console.log('Finished populating feedback content');
}

// ====================== EMAIL FUNCTIONALITY (EmailJS) ======================

async function sendEmailAutomatically(data) {
    try {
        // Wait a moment for chart to render
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Capture chart as image
        let chartImage = '';
        if (comparisonChart) {
            chartImage = comparisonChart.toBase64Image('image/png', 1.0);
        }
        
        // Prepare data
        const avgCloseness = data.userCloseness.reduce((a, b) => a + b, 0) / data.userCloseness.length;
        const feedbackText = data.feedback.join('\n\n');
        
        // Check if EmailJS is configured
        if (typeof emailjs === 'undefined' || EMAILJS_SERVICE_ID === 'YOUR_SERVICE_ID') {
            console.warn('EmailJS not configured. Please set up EmailJS credentials.');
            // Still show success message
            const emailSuccessSection = document.getElementById('emailSuccessSection');
            if (emailSuccessSection) {
                emailSuccessSection.style.display = 'block';
            }
            return;
        }
        
        // Build full HTML email content with embedded chart
        const htmlContent = `
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f5f5f5;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: white; border-radius: 10px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h1 style="color: #667eea; margin-top: 0; text-align: center;">🏀 ShotSync</h1>
            <h2 style="color: #333; text-align: center; font-size: 24px;">Thank You for Using ShotSync!</h2>
            
            <p style="font-size: 16px;">Hi ${userInfo.firstName},</p>
            <p>We're excited to share your personalized shot analysis results with you. Here's everything you need to improve your basketball shooting form:</p>
            
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 10px; margin: 30px 0; text-align: center;">
                <h2 style="color: white; margin: 0; font-size: 36px;">Overall Score: ${avgCloseness.toFixed(1)}%</h2>
            </div>
            
            ${chartImage ? `
            <h3 style="color: #333; margin-top: 30px;">Your Shot Analysis Graph</h3>
            <p style="color: #666; font-size: 14px;">This graph shows how your shot form compares to the benchmark throughout your shooting motion.</p>
            <img src="${chartImage}" alt="Shot Analysis Chart" style="max-width: 100%; height: auto; border-radius: 10px; margin: 20px 0; border: 2px solid #e0e0e0;">
            ` : ''}
            
            <h3 style="color: #333; margin-top: 30px;">Feedback & Recommendations</h3>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #667eea;">
                <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; margin: 0; color: #333;">${feedbackText}</pre>
            </div>
            
            <div style="background: #f0f7ff; padding: 20px; border-radius: 10px; margin: 30px 0; text-align: center;">
                <p style="margin: 0; font-size: 16px; color: #333;"><strong>Keep practicing!</strong> Every shot is an opportunity to improve.</p>
            </div>
            
            <p style="margin-top: 30px; color: #666; font-size: 14px;">We hope this analysis helps you take your game to the next level!</p>
            <p style="margin-top: 20px;">Best regards,<br><strong>The ShotSync Team</strong></p>
            
            <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">
            <p style="text-align: center; color: #999; font-size: 12px;">ShotSync - Your Basketball Shot Analysis Partner</p>
        </div>
    </div>
</body>
</html>
        `.trim();
        
        // Prepare email template parameters
        const templateParams = {
            to_email: userInfo.email,
            to_name: userInfo.firstName,
            subject: `Your ShotSync Analysis Results - ${userInfo.firstName}!`,
            message: htmlContent,
            html_content: htmlContent
        };
        
        // Send email via EmailJS using send method with HTML content
        await emailjs.send(EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, templateParams);
        
        // Save analysis to Firestore
        if (window.saveAnalysis && window.firebaseAuth?.currentUser) {
            try {
                const userId = window.firebaseAuth.currentUser.uid;
                await window.saveAnalysis(userId, {
                    firstName: userInfo.firstName,
                    lastName: userInfo.lastName,
                    email: userInfo.email,
                    overallScore: avgCloseness.toFixed(1),
                    selectedPlayer: selectedPlayer || 'custom',
                    feedback: feedbackText,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                console.error('Error saving analysis:', error);
                // Continue anyway - analysis saving is optional
            }
        }
        
        // Show success message
        const emailSuccessSection = document.getElementById('emailSuccessSection');
        if (emailSuccessSection) {
            emailSuccessSection.style.display = 'block';
        }
    } catch (error) {
        console.error('Error sending email:', error);
        // Still show success message (email might be delayed)
        const emailSuccessSection = document.getElementById('emailSuccessSection');
        if (emailSuccessSection) {
            emailSuccessSection.style.display = 'block';
        }
    }
}

// ====================== INITIALIZATION ======================

document.addEventListener('DOMContentLoaded', () => {
    initializePose();
    
    // Ensure step 0.5 (player selection) is shown by default
    const step0 = document.getElementById('step0');
    const step0_5 = document.getElementById('step0_5');
    if (step0 && step0_5) {
        step0.classList.remove('active');
        step0.style.display = 'none';
        step0_5.classList.add('active');
        step0_5.style.display = 'block';
    }
    
    // Google Sign-In button handlers (both on step0 and player selection page)
    const googleSignInBtn = document.getElementById('googleSignInBtn');
    if (googleSignInBtn && window.signInWithGoogle) {
        googleSignInBtn.addEventListener('click', handleGoogleSignIn);
    }

    const playerGoogleSignInBtn = document.getElementById('playerGoogleSignInBtn');
    if (playerGoogleSignInBtn && window.signInWithGoogle) {
        playerGoogleSignInBtn.addEventListener('click', handlePlayerPageSignIn);
    }
    
    // Profile dropdown toggle
    const profileButton = document.getElementById('profileButton');
    const profileMenu = document.getElementById('profileMenu');
    if (profileButton) {
        profileButton.addEventListener('click', (e) => {
            e.stopPropagation();
            profileMenu.style.display = profileMenu.style.display === 'none' ? 'block' : 'none';
        });
    }
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (profileMenu && profileButton) {
            const clickedInside = profileButton.contains(e.target) || profileMenu.contains(e.target);
            if (!clickedInside) {
                profileMenu.style.display = 'none';
            }
        }
    });
    
    // Logout button
    const logoutButton = document.getElementById('logoutButton');
    if (logoutButton) {
        logoutButton.addEventListener('click', handleLogout);
        logoutButton.addEventListener('mouseenter', () => {
            logoutButton.style.background = '#f5f5f5';
        });
        logoutButton.addEventListener('mouseleave', () => {
            logoutButton.style.background = 'none';
        });
    }
    
    // Check if user is already signed in
    if (window.onAuthStateChangedHandler && window.firebaseAuth) {
        window.onAuthStateChangedHandler(window.firebaseAuth, (user) => {
            const step0 = document.getElementById('step0');
            const step0_5 = document.getElementById('step0_5');
            
            if (user) {
                // User is signed in, update profile and show it
                updateProfileUI(user);
                
                const firstName = user.displayName?.split(' ')[0] || '';
                const lastName = user.displayName?.split(' ').slice(1).join(' ') || '';
                const email = user.email || '';
                
                // Store user info
                userInfo = { firstName, lastName, email };
                
                // Hide sign-in button since user is already signed in
                const signInSection = document.getElementById('signInSection');
                if (signInSection) {
                    signInSection.style.display = 'none';
                }

                // Enable player selection and hide sign-in section on player page
                const playerSelectionContainer = document.getElementById('playerSelectionContainer');
                if (playerSelectionContainer) {
                    playerSelectionContainer.style.opacity = '1';
                    playerSelectionContainer.style.pointerEvents = 'auto';
                }

                const playerSignInSection = document.getElementById('playerSignInSection');
                if (playerSignInSection) {
                    playerSignInSection.style.display = 'none';
                }
                
                // Auto-advance to player selection if user is already signed in
                if (step0 && step0_5) {
                    const currentActiveStep = document.querySelector('.step.active');
                    // Only auto-advance if we're on the landing page
                    if (currentActiveStep && currentActiveStep.id === 'step0') {
                        step0.classList.remove('active');
                        step0.style.display = 'none';
                        step0_5.classList.add('active');
                        step0_5.style.display = 'block';
                    }
                }
            } else {
                // User is signed out, hide profile but keep showing player selection
                hideProfileUI();

                // Show sign-in button
                const signInSection = document.getElementById('signInSection');
                if (signInSection) {
                    signInSection.style.display = 'block';
                }

                // Keep player selection visible even when not signed in
                if (step0 && step0_5) {
                    step0.classList.remove('active');
                    step0.style.display = 'none';
                    step0_5.classList.add('active');
                    step0_5.style.display = 'block';
                }
            }
        });
    } else {
        // Firebase not initialized or not available - show player selection page
        const step0 = document.getElementById('step0');
        const step0_5 = document.getElementById('step0_5');
        if (step0 && step0_5) {
            step0.classList.remove('active');
            step0.style.display = 'none';
            step0_5.classList.add('active');
            step0_5.style.display = 'block';
        }
    }
    
    document.getElementById('startBenchmark').addEventListener('click', startBenchmarkRecording);
    document.getElementById('stopBenchmark').addEventListener('click', stopBenchmarkRecording);
    document.getElementById('retakeBenchmark').addEventListener('click', retakeBenchmark);
    
    document.getElementById('startUser').addEventListener('click', startUserRecording);
    document.getElementById('stopUser').addEventListener('click', stopUserRecording);
    document.getElementById('retakeUser').addEventListener('click', retakeUser);
    
    document.getElementById('newComparison').addEventListener('click', resetApp);
    
    // Back to players buttons (using class since there are multiple)
    const backToPlayersButtons = document.querySelectorAll('.backToPlayers');
    backToPlayersButtons.forEach(backToPlayersBtn => {
    if (backToPlayersBtn) {
        backToPlayersBtn.addEventListener('click', () => {
            // Stop any active recordings
            if (benchmarkCamera) {
                benchmarkCamera.stop();
                benchmarkCamera = null;
            }
            if (userCamera) {
                userCamera.stop();
                userCamera = null;
            }
            if (benchmarkStream) {
                benchmarkStream.getTracks().forEach(track => track.stop());
                benchmarkStream = null;
            }
            if (userStream) {
                userStream.getTracks().forEach(track => track.stop());
                userStream = null;
            }
            
            // Hide all steps
            document.querySelectorAll('.step').forEach(step => {
                step.classList.remove('active');
                step.style.display = 'none';
            });
            
            // Show player selection
            const step0_5 = document.getElementById('step0_5');
            if (step0_5) {
                step0_5.classList.add('active');
                step0_5.style.display = 'block';
            }
            });
        }
    });
    
    // No longer need updateBackButton() function since buttons are always visible on recording pages
    
    // Initialize professional player benchmarks
    initializeProPlayerBenchmarks();
    
    // Set up player selection buttons
    const playerButtons = document.querySelectorAll('.player-btn');
    playerButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const player = btn.getAttribute('data-player');
            selectPlayer(player);
        });
    });
    
    // Initialize EmailJS if available
    if (typeof emailjs !== 'undefined' && EMAILJS_PUBLIC_KEY !== 'YOUR_PUBLIC_KEY') {
        emailjs.init(EMAILJS_PUBLIC_KEY);
    }

    // ====================== VIDEO UPLOAD MODE HANDLERS ======================

    // Toggle between record and upload modes
    const userRecordModeBtn = document.getElementById('userRecordModeBtn');
    const userUploadModeBtn = document.getElementById('userUploadModeBtn');
    const userRecordControls = document.getElementById('userRecordControls');
    const userUploadControls = document.getElementById('userUploadControls');

    if (userRecordModeBtn) {
        userRecordModeBtn.addEventListener('click', () => {
            userRecordControls.style.display = 'flex';
            userUploadControls.style.display = 'none';
            userRecordModeBtn.classList.add('active');
            userUploadModeBtn.classList.remove('active');

            // Stop any video playback and switch back to webcam if needed
            const video = document.getElementById('userVideo');
            if (video && !video.srcObject) {
                video.src = '';
                video.load();
            }
        });
    }

    if (userUploadModeBtn) {
        userUploadModeBtn.addEventListener('click', () => {
            userRecordControls.style.display = 'none';
            userUploadControls.style.display = 'flex';
            userUploadModeBtn.classList.add('active');
            userRecordModeBtn.classList.remove('active');

            // Stop any active recording
            if (userCamera) {
                userCamera.stop();
                userCamera = null;
            }
            if (userStream) {
                userStream.getTracks().forEach(track => track.stop());
                userStream = null;
            }
        });
    }

    // File selection handler
    const selectUserVideo = document.getElementById('selectUserVideo');
    const userVideoUpload = document.getElementById('userVideoUpload');
    const uploadedFileName = document.getElementById('uploadedFileName');
    const processUserVideo = document.getElementById('processUserVideo');

    if (selectUserVideo && userVideoUpload) {
        selectUserVideo.addEventListener('click', () => {
            userVideoUpload.click();
        });

        userVideoUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                uploadedFileName.textContent = `Selected: ${file.name}`;
                processUserVideo.style.display = 'inline-block';

                // Load video file into video element
                const video = document.getElementById('userVideo');
                const videoURL = URL.createObjectURL(file);
                video.src = videoURL;
                video.loop = false;
                video.muted = true;
                video.load();
            }
        });
    }

    // Process uploaded video handler
    if (processUserVideo) {
        processUserVideo.addEventListener('click', () => {
            processUploadedUserVideo();
        });
    }
});

// ====================== PLAYER SELECTION ======================

function selectPlayer(player) {
    // Back buttons are now always visible on recording pages, no need to manually show them
    selectedPlayer = player;
    
    // Hide player selection (step0_5)
    const step0_5 = document.getElementById('step0_5');
    if (step0_5) {
        step0_5.classList.remove('active');
        step0_5.style.display = 'none';
    }
    
    // Show custom explanation if custom selected
    const customExplanation = document.getElementById('customExplanation');
    if (customExplanation) {
        if (player === 'custom') {
            customExplanation.style.display = 'block';
        } else {
            customExplanation.style.display = 'none';
        }
    }
    
    const playerNames = {
        'curry': 'Stephen Curry',
        'lebron': 'LeBron James',
        'jordan': 'Michael Jordan',
        'durant': 'Kevin Durant',
        'clark': 'Caitlin Clark'
    };
    
    if (player === 'custom') {
        // Custom mode: show benchmark recording step
        const step1Title = document.getElementById('step1Title');
        if (step1Title) {
            step1Title.textContent = 'Step 1: Record Benchmark Shot';
        }
        document.getElementById('step1').classList.add('active');
        document.getElementById('step1').style.display = 'block';
    } else {
        // Pro player mode: skip to user recording with pre-loaded benchmark
        // Use the pre-loaded benchmark data
        if (!proPlayerBenchmarks[player] || proPlayerBenchmarks[player].length === 0) {
            // Initialize if not already done
            initializeProPlayerBenchmarks();
        }
        
        const step2Title = document.getElementById('step2Title');
        if (step2Title) {
            step2Title.textContent = `Record Your Shot (vs ${playerNames[player]})`;
        }
        document.getElementById('step2').classList.add('active');
        document.getElementById('step2').style.display = 'block';
    }
}

// ====================== USER INFO COLLECTION ======================

async function handlePlayerPageSignIn() {
    try {
        const playerGoogleSignInBtn = document.getElementById('playerGoogleSignInBtn');
        if (playerGoogleSignInBtn) {
            playerGoogleSignInBtn.disabled = true;
            playerGoogleSignInBtn.textContent = 'Signing in...';
        }

        if (!window.signInWithGoogle || typeof window.signInWithGoogle !== 'function') {
            throw new Error('Sign-in function is not available. Please refresh the page and try again.');
        }

        const userData = await window.signInWithGoogle();

        // Store user info
        userInfo = userData;

        // Update profile UI on recording pages
        updateRecordingPageProfile(window.firebaseAuth?.currentUser);

        // Save to Firestore
        if (window.saveUserEmail && window.firebaseAuth?.currentUser) {
            await window.saveUserEmail(userData.email, userData.firstName, userData.lastName);
        }
        
        // Update login streak
        if (window.updateLoginStreak && window.firebaseAuth?.currentUser) {
            await window.updateLoginStreak(window.firebaseAuth.currentUser.uid);
        }

        // Enable player selection
        const playerSelectionContainer = document.getElementById('playerSelectionContainer');
        if (playerSelectionContainer) {
            playerSelectionContainer.style.opacity = '1';
            playerSelectionContainer.style.pointerEvents = 'auto';
        }

        // Hide sign-in section
        const playerSignInSection = document.getElementById('playerSignInSection');
        if (playerSignInSection) {
            playerSignInSection.style.display = 'none';
        }
    } catch (error) {
        console.error('Error signing in with Google:', error);
        console.error('Error code:', error.code);
        console.error('Error message:', error.message);

        let errorMessage = 'Failed to sign in with Google.\n\n';
        if (error.code === 'auth/operation-not-allowed') {
            errorMessage += 'Google Sign-In is not enabled in Firebase. Please enable it in Firebase Console under Authentication > Sign-in method.';
        } else if (error.code === 'auth/unauthorized-domain') {
            errorMessage += 'This domain is not authorized. Please add this domain to Firebase authorized domains.';
        } else if (error.code === 'auth/popup-blocked') {
            errorMessage += 'Popup was blocked by your browser. Please allow popups for this site and try again.';
        } else if (error.code === 'auth/popup-closed-by-user') {
            errorMessage += 'Sign-in popup was closed. Please try again.';
        } else if (error.code === 'auth/cancelled-popup-request') {
            errorMessage += 'Another sign-in attempt is already in progress. Please wait and try again.';
        } else if (error.code) {
            errorMessage += `Error code: ${error.code}\nError message: ${error.message || 'Unknown error'}`;
        } else {
            errorMessage += `Error: ${error.message || 'Unknown error occurred. Please check the browser console for details.'}`;
        }

        alert(errorMessage);

        const playerGoogleSignInBtn = document.getElementById('playerGoogleSignInBtn');
        if (playerGoogleSignInBtn) {
            playerGoogleSignInBtn.disabled = false;
            playerGoogleSignInBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 18 18"><path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z"/><path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332C2.438 15.983 5.482 18 9 18z"/><path fill="#FBBC05" d="M3.964 10.712c-.18-.54-.282-1.117-.282-1.712 0-.595.102-1.172.282-1.712V4.956H.957C.348 6.175 0 7.55 0 9s.348 2.825.957 4.044l3.007-2.332z"/><path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0 5.482 0 2.438 2.017.957 4.956L3.964 7.288C4.672 5.163 6.656 3.58 9 3.58z"/></svg> Sign in with Google';
        }
    }
}

async function handleGoogleSignIn() {
    try {
        const googleSignInBtn = document.getElementById('googleSignInBtn');
        if (googleSignInBtn) {
            googleSignInBtn.disabled = true;
            googleSignInBtn.textContent = 'Signing in...';
        }
        
        if (!window.signInWithGoogle || typeof window.signInWithGoogle !== 'function') {
            throw new Error('Sign-in function is not available. Please refresh the page and try again.');
        }
        
        const userData = await window.signInWithGoogle();
        
        // Store user info
        userInfo = userData;
        
        // Update profile UI
        if (window.firebaseAuth?.currentUser) {
            updateProfileUI(window.firebaseAuth.currentUser);
        }
        
        // Save to Firestore
        if (window.saveUserEmail && window.firebaseAuth?.currentUser) {
            await window.saveUserEmail(userData.email, userData.firstName, userData.lastName);
        }
        
        // Update login streak
        if (window.updateLoginStreak && window.firebaseAuth?.currentUser) {
            await window.updateLoginStreak(window.firebaseAuth.currentUser.uid);
        }
    
    // Move to player selection step
    document.getElementById('step0').classList.remove('active');
    document.getElementById('step0').style.display = 'none';
    document.getElementById('step0_5').classList.add('active');
    document.getElementById('step0_5').style.display = 'block';
    } catch (error) {
        console.error('Error signing in with Google:', error);
        console.error('Error code:', error.code);
        console.error('Error message:', error.message);
        
        // Provide more specific error messages
        let errorMessage = 'Failed to sign in with Google.\n\n';
        if (error.code === 'auth/operation-not-allowed') {
            errorMessage += 'Google Sign-In is not enabled in Firebase. Please enable it in Firebase Console under Authentication > Sign-in method.';
        } else if (error.code === 'auth/unauthorized-domain') {
            errorMessage += 'This domain is not authorized. Please add this domain to Firebase authorized domains.';
        } else if (error.code === 'auth/popup-blocked') {
            errorMessage += 'Popup was blocked by your browser. Please allow popups for this site and try again.';
        } else if (error.code === 'auth/popup-closed-by-user') {
            errorMessage += 'Sign-in popup was closed. Please try again.';
        } else if (error.code === 'auth/cancelled-popup-request') {
            errorMessage += 'Another sign-in attempt is already in progress. Please wait and try again.';
        } else if (error.code) {
            errorMessage += `Error code: ${error.code}\nError message: ${error.message || 'Unknown error'}`;
        } else {
            errorMessage += `Error: ${error.message || 'Unknown error occurred. Please check the browser console for details.'}`;
        }
        
        alert(errorMessage);
        
        const googleSignInBtn = document.getElementById('googleSignInBtn');
        if (googleSignInBtn) {
            googleSignInBtn.disabled = false;
            googleSignInBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 18 18"><path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z"/><path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332C2.438 15.983 5.482 18 9 18z"/><path fill="#FBBC05" d="M3.964 10.712c-.18-.54-.282-1.117-.282-1.712 0-.595.102-1.172.282-1.712V4.956H.957C.348 6.175 0 7.55 0 9s.348 2.825.957 4.044l3.007-2.332z"/><path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0 5.482 0 2.438 2.017.957 4.956L3.964 7.288C4.672 5.163 6.656 3.58 9 3.58z"/></svg> Sign in with Google';
        }
    }
}

// ====================== PROFILE MANAGEMENT ======================

function updateRecordingPageProfile(user) {
    if (!user) return;

    const displayName = user.displayName || 'User';
    const firstName = displayName.split(' ')[0] || 'User';
    const lastName = displayName.split(' ').slice(1).join(' ') || '';
    const initials = ((firstName[0] || '') + (lastName[0] || '')).toUpperCase() || 'U';
    const email = user.email || '';

    // Update step1 profile
    const step1Display = document.getElementById('step1ProfileDisplay');
    const step1Initials = document.getElementById('step1ProfileInitials');
    const step1Name = document.getElementById('step1ProfileName');
    const step1Email = document.getElementById('step1ProfileEmail');

    if (step1Display) step1Display.style.display = 'block';
    if (step1Initials) step1Initials.textContent = initials;
    if (step1Name) step1Name.textContent = displayName;
    if (step1Email) step1Email.textContent = email;

    // Update step2 profile
    const step2Display = document.getElementById('step2ProfileDisplay');
    const step2Initials = document.getElementById('step2ProfileInitials');
    const step2Name = document.getElementById('step2ProfileName');
    const step2Email = document.getElementById('step2ProfileEmail');

    if (step2Display) step2Display.style.display = 'block';
    if (step2Initials) step2Initials.textContent = initials;
    if (step2Name) step2Name.textContent = displayName;
    if (step2Email) step2Email.textContent = email;
}

function updateProfileUI(user) {
    const profileDropdown = document.getElementById('profileDropdown');
    const profileInitials = document.getElementById('profileInitials');
    const profileName = document.getElementById('profileName');
    const menuUserName = document.getElementById('menuUserName');
    const menuUserEmail = document.getElementById('menuUserEmail');
    
    if (!profileDropdown) return;
    
    // Show profile dropdown
    profileDropdown.style.display = 'block';
    
    // Get user name
    const displayName = user.displayName || 'User';
    const firstName = displayName.split(' ')[0] || 'User';
    const lastName = displayName.split(' ').slice(1).join(' ') || '';
    
    // Set initials
    const initials = (firstName[0] || '') + (lastName[0] || '') || 'U';
    if (profileInitials) {
        profileInitials.textContent = initials.toUpperCase();
    }
    
    // Set name
    if (profileName) {
        profileName.textContent = displayName;
    }
    
    // Set menu info
    if (menuUserName) {
        menuUserName.textContent = displayName;
    }
    if (menuUserEmail) {
        menuUserEmail.textContent = user.email || '';
    }

    // Also update recording page profiles
    updateRecordingPageProfile(user);
}

function hideProfileUI() {
    const profileDropdown = document.getElementById('profileDropdown');
    if (profileDropdown) {
        profileDropdown.style.display = 'none';
    }
}

async function handleLogout() {
    try {
        if (window.signOutUser) {
            await window.signOutUser();
        }
        
        // Clear user info
        userInfo = null;
        selectedPlayer = null;
        
        // Hide profile
        hideProfileUI();
        
        // Reset app and go back to step 0
        resetApp();
        
        // Show success message
        alert('Signed out successfully');
    } catch (error) {
        console.error('Error signing out:', error);
        alert('Failed to sign out. Please try again.');
    }
}

function retakeBenchmark() {
    benchmarkPoseData = [];
    document.getElementById('retakeBenchmark').style.display = 'none';
    document.getElementById('benchmarkStatus').textContent = '';
    document.getElementById('benchmarkStatus').className = 'status';
}

function retakeUser() {
    userPoseData = [];
    document.getElementById('retakeUser').style.display = 'none';
    document.getElementById('userStatus').textContent = '';
    document.getElementById('userStatus').className = 'status';
}

function resetApp() {
    if (benchmarkCamera) benchmarkCamera.stop();
    if (userCamera) userCamera.stop();
    if (benchmarkStream) benchmarkStream.getTracks().forEach(track => track.stop());
    if (userStream) userStream.getTracks().forEach(track => track.stop());
    
    benchmarkPoseData = [];
    userPoseData = [];
    benchmarkCamera = null;
    userCamera = null;
    benchmarkStream = null;
    userStream = null;
    selectedPlayer = null;
    // Don't clear userInfo - keep user signed in
    
    // Hide all steps first
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active');
        step.style.display = 'none';
    });
    
    // Reset UI - go back to player selection if signed in, otherwise landing page
    const step0 = document.getElementById('step0');
    const step0_5 = document.getElementById('step0_5');
    
    // Check if user is signed in - check Firebase auth state synchronously
    let isSignedIn = false;
    if (window.firebaseAuth && window.firebaseAuth.currentUser) {
        isSignedIn = true;
    } else if (userInfo !== null) {
        isSignedIn = true;
    }
    
    // Always go to player selection, regardless of sign-in status
    if (step0) {
        step0.classList.remove('active');
        step0.style.display = 'none';
    }
    if (step0_5) {
        step0_5.classList.add('active');
        step0_5.style.display = 'block';
    }
    
    document.getElementById('step1').classList.remove('active');
    document.getElementById('step1').style.display = 'none';
    document.getElementById('step2').classList.remove('active');
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step3').classList.remove('active');
    document.getElementById('step3').style.display = 'none';
    
    document.getElementById('startBenchmark').disabled = false;
    document.getElementById('stopBenchmark').disabled = true;
    document.getElementById('startUser').disabled = false;
    document.getElementById('stopUser').disabled = true;
    
    document.getElementById('retakeBenchmark').style.display = 'none';
    document.getElementById('retakeUser').style.display = 'none';
    
    const emailSuccessSection = document.getElementById('emailSuccessSection');
    if (emailSuccessSection) {
        emailSuccessSection.style.display = 'none';
    }
    
    if (comparisonChart) {
        comparisonChart.destroy();
        comparisonChart = null;
    }
}

