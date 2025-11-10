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
    const fps = 30; // 30 frames per second  
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
            
            if (results.poseLandmarks) {
                drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
                    color: '#00FF00',
                    lineWidth: 2
                });
                drawLandmarks(ctx, results.poseLandmarks, {
                    color: '#00FF00',
                    lineWidth: 1,
                    radius: 3
                });
                
                const state = getArmState(results.poseLandmarks, canvas.width, canvas.height);
                const currentTime = Date.now() / 1000.0;
                
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
        
        // Update back button visibility
        const backBtn = document.getElementById('backToPlayers');
        if (backBtn) backBtn.style.display = 'block';
    }
}

async function startUserRecording() {
    try {
        const video = document.getElementById('userVideo');
        const canvas = document.getElementById('userOutput');
        const ctx = canvas.getContext('2d');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        userStream = stream;
        video.srcObject = stream;
        
        // Set canvas dimensions
        canvas.width = 640;
        canvas.height = 480;
        
        // Ensure video plays (non-blocking)
        video.play().catch(err => console.error('Video play error:', err));
        
        userPoseData = [];
        
        let previousStage = "neutral";
        let startTime = null;
        let recordingActive = false;
        let seenFollowThrough = false;
        const lastPrintTime = { value: Date.now() };
        
        document.getElementById('startUser').disabled = true;
        document.getElementById('stopUser').disabled = false;
        document.getElementById('userStatus').textContent = 'Recording...';
        document.getElementById('userStatus').className = 'status recording';
        
        userPose.onResults((results) => {
            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw the video frame
            if (results.image) {
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
            }
            
            if (results.poseLandmarks) {
                drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
                    color: '#00FF00',
                    lineWidth: 2
                });
                drawLandmarks(ctx, results.poseLandmarks, {
                    color: '#00FF00',
                    lineWidth: 1,
                    radius: 3
                });
                
                const state = getArmState(results.poseLandmarks, canvas.width, canvas.height);
                const currentTime = Date.now() / 1000.0;
                
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

function compareShots() {
    document.getElementById('step2').classList.remove('active');
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step3').classList.add('active');
    document.getElementById('step3').style.display = 'block';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    setTimeout(() => {
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
            playerName: selectedPlayer
        });
    }, 500);
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
    
    comparisonChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.userTimes.map(t => t.toFixed(2)),
            datasets: [{
                label: 'Benchmark (100%)',
                data: data.benchTimes.map(() => 100),
                borderColor: 'rgb(255, 159, 64)',
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0
            }, {
                label: 'Your Shot',
                data: data.userCloseness,
                borderColor: 'rgb(54, 162, 235)',
                borderWidth: 2,
                fill: false,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Shot Form Analysis'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 110,
                    title: {
                        display: true,
                        text: 'Closeness to Benchmark (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                }
            }
        }
    });
    
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

function generatePlayerSpecificFeedback(data) {
    const player = data.playerName;
    if (!player || player === 'custom') {
        return generateGenericFeedback(data);
    }
    
    // Calculate key metrics from the data
    const avgCloseness = data.userCloseness.reduce((a, b) => a + b, 0) / data.userCloseness.length;
    
    // Extract angle data from userPoseData (use global variable)
    let avgElbowAngle = 0;
    let avgWristAngle = 0;
    let avgArmAngle = 0;
    let elbowCount = 0;
    let wristCount = 0;
    let armCount = 0;
    
    // Try to get userPoseData from global scope
    const poseData = typeof userPoseData !== 'undefined' ? userPoseData : [];
    
    if (poseData && poseData.length > 0) {
        poseData.forEach(frame => {
            if (frame && frame.elbow_angle !== null && frame.elbow_angle !== undefined && !isNaN(frame.elbow_angle)) {
                avgElbowAngle += frame.elbow_angle;
                elbowCount++;
            }
            if (frame && frame.wrist_angle !== null && frame.wrist_angle !== undefined && !isNaN(frame.wrist_angle)) {
                avgWristAngle += frame.wrist_angle;
                wristCount++;
            }
            if (frame && frame.arm_angle !== null && frame.arm_angle !== undefined && !isNaN(frame.arm_angle)) {
                avgArmAngle += frame.arm_angle;
                armCount++;
            }
        });
        
        if (elbowCount > 0) avgElbowAngle /= elbowCount;
        if (wristCount > 0) avgWristAngle /= wristCount;
        if (armCount > 0) avgArmAngle /= armCount;
    }
    
    // Use default values if no data available
    if (elbowCount === 0) avgElbowAngle = 145;
    if (wristCount === 0) avgWristAngle = 90;
    if (armCount === 0) avgArmAngle = 50;
    
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
        // High release point: optimal arm angle
        nicheScore = (avgArmAngle < 50 ? (avgArmAngle / 50 * 100) : 100) * 0.7 + avgCloseness * 0.3;
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
        { label: 'Elbow Angle', value: `${avgElbowAngle.toFixed(1)}°`, ideal: `${feedback.idealElbow}°`, score: Math.max(0, 100 - elbowDiff * 2) },
        { label: 'Wrist Angle', value: `${avgWristAngle.toFixed(1)}°`, ideal: `${feedback.idealWrist}°`, score: Math.max(0, 100 - wristDiff * 3) },
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
                        <span class="feedback-value">Your: ${strength.value}</span>
                        <span class="feedback-ideal">Ideal: ${strength.ideal}</span>
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
                        <span class="feedback-value">Your: ${weakness.value}</span>
                        <span class="feedback-ideal">Ideal: ${weakness.ideal}</span>
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
            <h1 style="color: #667eea; margin-top: 0; text-align: center;">🏀 Shot Sync</h1>
            <h2 style="color: #333; text-align: center; font-size: 24px;">Thank You for Using Shot Sync!</h2>
            
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
            <p style="margin-top: 20px;">Best regards,<br><strong>The Shot Sync Team</strong></p>
            
            <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">
            <p style="text-align: center; color: #999; font-size: 12px;">Shot Sync - Your Basketball Shot Analysis Partner</p>
        </div>
    </div>
</body>
</html>
        `.trim();
        
        // Prepare email template parameters
        const templateParams = {
            to_email: userInfo.email,
            to_name: userInfo.firstName,
            subject: `Your Shot Sync Analysis Results - ${userInfo.firstName}!`,
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
    
    // Ensure step 0 (landing page) is shown by default
    const step0 = document.getElementById('step0');
    const step0_5 = document.getElementById('step0_5');
    if (step0 && step0_5) {
        step0.classList.add('active');
        step0.style.display = 'block';
        step0_5.classList.remove('active');
        step0_5.style.display = 'none';
    }
    
    // Google Sign-In button handler
    const googleSignInBtn = document.getElementById('googleSignInBtn');
    if (googleSignInBtn && window.signInWithGoogle) {
        googleSignInBtn.addEventListener('click', handleGoogleSignIn);
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
                // User is signed out, hide profile and show landing page
                hideProfileUI();
                
                // Show sign-in button
                const signInSection = document.getElementById('signInSection');
                if (signInSection) {
                    signInSection.style.display = 'block';
                }
                
                if (step0 && step0_5) {
                    step0.classList.add('active');
                    step0.style.display = 'block';
                    step0_5.classList.remove('active');
                    step0_5.style.display = 'none';
                }
            }
        });
    } else {
        // Firebase not initialized or not available - show landing page
        const step0 = document.getElementById('step0');
        const step0_5 = document.getElementById('step0_5');
        if (step0 && step0_5) {
            step0.classList.add('active');
            step0.style.display = 'block';
            step0_5.classList.remove('active');
            step0_5.style.display = 'none';
        }
    }
    
    document.getElementById('startBenchmark').addEventListener('click', startBenchmarkRecording);
    document.getElementById('stopBenchmark').addEventListener('click', stopBenchmarkRecording);
    document.getElementById('retakeBenchmark').addEventListener('click', retakeBenchmark);
    
    document.getElementById('startUser').addEventListener('click', startUserRecording);
    document.getElementById('stopUser').addEventListener('click', stopUserRecording);
    document.getElementById('retakeUser').addEventListener('click', retakeUser);
    
    document.getElementById('newComparison').addEventListener('click', resetApp);
    
    // Back to players button
    const backToPlayersBtn = document.getElementById('backToPlayers');
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
            
            // Hide back button
            backToPlayersBtn.style.display = 'none';
        });
    }
    
    // Show/hide back button based on current step
    function updateBackButton() {
        const step1 = document.getElementById('step1');
        const step2 = document.getElementById('step2');
        const backBtn = document.getElementById('backToPlayers');
        if (backBtn) {
            const isOnRecordingPage = (step1 && step1.style.display !== 'none') || 
                                     (step2 && step2.style.display !== 'none');
            backBtn.style.display = isOnRecordingPage ? 'block' : 'none';
        }
    }
    
    // Monitor step changes
    const observer = new MutationObserver(updateBackButton);
    document.querySelectorAll('.step').forEach(step => {
        observer.observe(step, { attributes: true, attributeFilter: ['style', 'class'] });
    });
    updateBackButton(); // Initial check
    
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
});

// ====================== PLAYER SELECTION ======================

function selectPlayer(player) {
    // Show back button when navigating to recording pages
    const backBtn = document.getElementById('backToPlayers');
    if (backBtn) backBtn.style.display = 'block';
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

async function handleGoogleSignIn() {
    try {
        const googleSignInBtn = document.getElementById('googleSignInBtn');
        if (googleSignInBtn) {
            googleSignInBtn.disabled = true;
            googleSignInBtn.textContent = 'Signing in...';
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
    
    // Move to player selection step
    document.getElementById('step0').classList.remove('active');
    document.getElementById('step0').style.display = 'none';
    document.getElementById('step0_5').classList.add('active');
    document.getElementById('step0_5').style.display = 'block';
    } catch (error) {
        console.error('Error signing in with Google:', error);
        
        // Provide more specific error messages
        let errorMessage = 'Failed to sign in with Google. ';
        if (error.code === 'auth/operation-not-allowed') {
            errorMessage += 'Google Sign-In is not enabled in Firebase. Please enable it in Firebase Console.';
        } else if (error.code === 'auth/unauthorized-domain') {
            errorMessage += 'This domain is not authorized. Please add shubh-go.github.io to Firebase authorized domains.';
        } else {
            errorMessage += 'Please try again or use the form below.';
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
    
    if (isSignedIn) {
        // User is signed in - go to player selection
        if (step0) {
            step0.classList.remove('active');
            step0.style.display = 'none';
        }
        if (step0_5) {
            step0_5.classList.add('active');
            step0_5.style.display = 'block';
        }
    } else {
        // User is not signed in - go to landing page
    if (step0) {
        step0.classList.add('active');
        step0.style.display = 'block';
        const userInfoForm = document.getElementById('userInfoForm');
        if (userInfoForm) {
            userInfoForm.reset();
        }
    }
    if (step0_5) {
        step0_5.classList.remove('active');
        step0_5.style.display = 'none';
        }
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

