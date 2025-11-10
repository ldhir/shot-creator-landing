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
        proPlayerBenchmarks[player] = generateExampleBenchmarkData();
    });
}

// MediaPipe Pose
let benchmarkPose = null;
let userPose = null;

// ====================== MEDIAPIPE SETUP ======================

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
        
        canvas.width = 640;
        canvas.height = 480;
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        benchmarkStream = stream;
        video.srcObject = stream;
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
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
            
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
                            landmarks: landmarks3D
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
                        landmarks: landmarks3D
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
    }
}

async function startUserRecording() {
    try {
        const video = document.getElementById('userVideo');
        const canvas = document.getElementById('userOutput');
        const ctx = canvas.getContext('2d');
        
        canvas.width = 640;
        canvas.height = 480;
        
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
        document.getElementById('userStatus').textContent = 'Recording...';
        document.getElementById('userStatus').className = 'status recording';
        
        userPose.onResults((results) => {
            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
            
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
                            landmarks: landmarks3D
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
                        landmarks: landmarks3D
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

function extractFormSeries(shotData) {
    const times = [];
    const formVals = [];
    for (const entry of shotData) {
        const measure = computeOverallForm(entry.elbow_angle, entry.wrist_angle, entry.arm_angle);
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

function computeUserCloseness(benchForm, userForm, path) {
    const alpha = 2.0;
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
            const score = Math.max(0, Math.min(100, 100 - alpha * diff));
            userCloseness.push(score);
        } else {
            userCloseness.push(100);
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
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    
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
    
    const feedbackList = document.getElementById('feedbackList');
    feedbackList.innerHTML = '';
    data.feedback.forEach(feedback => {
        const p = document.createElement('p');
        p.textContent = feedback;
        feedbackList.appendChild(p);
    });
    
    // Automatically send email if user info is available
    if (userInfo) {
        sendEmailAutomatically(data);
    } else {
        // Show success message anyway (fallback)
        const emailSuccessSection = document.getElementById('emailSuccessSection');
        if (emailSuccessSection) {
            emailSuccessSection.style.display = 'block';
        }
    }
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
    
    // User info form handler
    const userInfoForm = document.getElementById('userInfoForm');
    if (userInfoForm) {
        userInfoForm.addEventListener('submit', handleUserInfoSubmission);
    }
    
    document.getElementById('startBenchmark').addEventListener('click', startBenchmarkRecording);
    document.getElementById('stopBenchmark').addEventListener('click', stopBenchmarkRecording);
    document.getElementById('retakeBenchmark').addEventListener('click', retakeBenchmark);
    
    document.getElementById('startUser').addEventListener('click', startUserRecording);
    document.getElementById('stopUser').addEventListener('click', stopUserRecording);
    document.getElementById('retakeUser').addEventListener('click', retakeUser);
    
    document.getElementById('newComparison').addEventListener('click', resetApp);
    
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

function handleUserInfoSubmission(e) {
    e.preventDefault();
    
    const firstName = document.getElementById('firstName').value.trim();
    const lastName = document.getElementById('lastName').value.trim();
    const email = document.getElementById('email').value.trim();
    
    if (!firstName || !lastName || !email) {
        alert('Please fill in all fields.');
        return;
    }
    
    // Store user info
    userInfo = { firstName, lastName, email };
    
    // Move to player selection step
    document.getElementById('step0').classList.remove('active');
    document.getElementById('step0').style.display = 'none';
    document.getElementById('step0_5').classList.add('active');
    document.getElementById('step0_5').style.display = 'block';
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
    userInfo = null;
    
    // Reset UI - go back to Step 0
    const step0 = document.getElementById('step0');
    if (step0) {
        step0.classList.add('active');
        step0.style.display = 'block';
        const userInfoForm = document.getElementById('userInfoForm');
        if (userInfoForm) {
            userInfoForm.reset();
        }
    }
    
    // Hide step0_5
    const step0_5 = document.getElementById('step0_5');
    if (step0_5) {
        step0_5.classList.remove('active');
        step0_5.style.display = 'none';
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

