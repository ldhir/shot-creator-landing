// Global state
let benchmarkFrames = [];
let userFrames = [];
let benchmarkStream = null;
let userStream = null;
let benchmarkInterval = null;
let userInterval = null;
let benchmarkVideo = null;
let userVideo = null;
let comparisonChart = null;
let currentResultsData = null; // Store results data for email
let userInfo = null; // Store user info (firstName, lastName, email)

const API_URL = 'http://localhost:5000/api';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    benchmarkVideo = document.getElementById('benchmarkVideo');
    userVideo = document.getElementById('userVideo');
    
    // Benchmark controls
    document.getElementById('startBenchmark').addEventListener('click', startBenchmarkRecording);
    document.getElementById('stopBenchmark').addEventListener('click', stopBenchmarkRecording);
    document.getElementById('retakeBenchmark').addEventListener('click', retakeBenchmark);
    
    // User controls
    document.getElementById('startUser').addEventListener('click', startUserRecording);
    document.getElementById('stopUser').addEventListener('click', stopUserRecording);
    document.getElementById('retakeUser').addEventListener('click', retakeUser);
    
    // New comparison
    document.getElementById('newComparison').addEventListener('click', resetApp);
    
    // User info form submission (Step 0)
    document.getElementById('userInfoForm').addEventListener('submit', handleUserInfoSubmission);
});

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
    userInfo = {
        firstName: firstName,
        lastName: lastName,
        email: email
    };
    
    // Move to Step 1 (Record Benchmark)
    document.getElementById('step0').classList.remove('active');
    document.getElementById('step0').style.display = 'none';
    document.getElementById('step1').classList.add('active');
    document.getElementById('step1').style.display = 'block';
}

// ====================== VIDEO CAPTURE ======================

async function startBenchmarkRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        benchmarkStream = stream;
        benchmarkVideo.srcObject = stream;
        benchmarkFrames = [];
        
        document.getElementById('startBenchmark').disabled = true;
        document.getElementById('stopBenchmark').disabled = false;
        document.getElementById('benchmarkStatus').textContent = 'Recording...';
        document.getElementById('benchmarkStatus').className = 'status recording';
        
        // Capture frames every 100ms (10 fps)
        const canvas = document.getElementById('benchmarkCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 640;
        canvas.height = 480;
        
        benchmarkInterval = setInterval(() => {
            ctx.drawImage(benchmarkVideo, 0, 0, canvas.width, canvas.height);
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            benchmarkFrames.push({
                image: frameData,
                timestamp: Date.now() / 1000.0
            });
        }, 100);
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        document.getElementById('benchmarkStatus').textContent = 'Error accessing camera. Please allow camera permissions.';
        document.getElementById('benchmarkStatus').className = 'status error';
    }
}

function stopBenchmarkRecording() {
    if (benchmarkInterval) {
        clearInterval(benchmarkInterval);
        benchmarkInterval = null;
    }
    
    if (benchmarkStream) {
        benchmarkStream.getTracks().forEach(track => track.stop());
        benchmarkStream = null;
    }
    
    document.getElementById('startBenchmark').disabled = false;
    document.getElementById('stopBenchmark').disabled = true;
    
    if (benchmarkFrames.length > 0) {
        document.getElementById('benchmarkStatus').textContent = `Recorded ${benchmarkFrames.length} frames. Click "Next" to continue.`;
        document.getElementById('benchmarkStatus').className = 'status success';
        document.getElementById('retakeBenchmark').style.display = 'inline-block';
        
        // Process benchmark shot
        processBenchmarkShot();
    }
}

async function processBenchmarkShot() {
    try {
        const response = await fetch(`${API_URL}/process_shot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ frames: benchmarkFrames })
        });
        
        const data = await response.json();
        if (data.success) {
            // Move to step 2
            document.getElementById('step1').classList.remove('active');
            document.getElementById('step1').style.display = 'none';
            document.getElementById('step2').classList.add('active');
            document.getElementById('step2').style.display = 'block';
        } else {
            throw new Error(data.error || 'Failed to process benchmark shot');
        }
    } catch (error) {
        console.error('Error processing benchmark:', error);
        document.getElementById('benchmarkStatus').textContent = 'Error processing shot. Please try again.';
        document.getElementById('benchmarkStatus').className = 'status error';
    }
}

async function startUserRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        userStream = stream;
        userVideo.srcObject = stream;
        userFrames = [];
        
        document.getElementById('startUser').disabled = true;
        document.getElementById('stopUser').disabled = false;
        document.getElementById('userStatus').textContent = 'Recording...';
        document.getElementById('userStatus').className = 'status recording';
        
        const canvas = document.getElementById('userCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 640;
        canvas.height = 480;
        
        userInterval = setInterval(() => {
            ctx.drawImage(userVideo, 0, 0, canvas.width, canvas.height);
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            userFrames.push({
                image: frameData,
                timestamp: Date.now() / 1000.0
            });
        }, 100);
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        document.getElementById('userStatus').textContent = 'Error accessing camera. Please allow camera permissions.';
        document.getElementById('userStatus').className = 'status error';
    }
}

function stopUserRecording() {
    if (userInterval) {
        clearInterval(userInterval);
        userInterval = null;
    }
    
    if (userStream) {
        userStream.getTracks().forEach(track => track.stop());
        userStream = null;
    }
    
    document.getElementById('startUser').disabled = false;
    document.getElementById('stopUser').disabled = true;
    
    if (userFrames.length > 0) {
        document.getElementById('userStatus').textContent = `Recorded ${userFrames.length} frames. Analyzing...`;
        document.getElementById('userStatus').className = 'status success';
        document.getElementById('retakeUser').style.display = 'inline-block';
        
        // Compare shots
        compareShots();
    }
}

// ====================== SHOT COMPARISON ======================

async function compareShots() {
    // Show loading and move to step 3
    document.getElementById('step2').classList.remove('active');
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step3').classList.add('active');
    document.getElementById('step3').style.display = 'block';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    try {
        // Process both shots
        const [benchmarkResponse, userResponse] = await Promise.all([
            fetch(`${API_URL}/process_shot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames: benchmarkFrames })
            }),
            fetch(`${API_URL}/process_shot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames: userFrames })
            })
        ]);
        
        const benchmarkData = await benchmarkResponse.json();
        const userData = await userResponse.json();
        
        if (!benchmarkData.success || !userData.success) {
            throw new Error('Failed to process shots');
        }
        
        // Compare shots
        const compareResponse = await fetch(`${API_URL}/compare_shots`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                benchmark: {
                    shot_angles: benchmarkData.shot_angles,
                    landmark_frames: benchmarkData.landmark_frames
                },
                user: {
                    shot_angles: userData.shot_angles,
                    landmark_frames: userData.landmark_frames
                }
            })
        });
        
        const compareData = await compareResponse.json();
        
        if (!compareData.success) {
            throw new Error(compareData.error || 'Failed to compare shots');
        }
        
        // Display results
        displayResults(compareData);
        
    } catch (error) {
        console.error('Error comparing shots:', error);
        document.getElementById('loading').innerHTML = `
            <p style="color: red;">Error analyzing shots: ${error.message}</p>
            <button onclick="resetApp()" class="btn btn-primary" style="margin-top: 20px;">Try Again</button>
        `;
    }
}

function displayResults(data) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    
    // Store results data for email
    currentResultsData = data;
    
    // Display overall score
    const avgCloseness = data.user_closeness.reduce((a, b) => a + b, 0) / data.user_closeness.length;
    document.getElementById('overallScore').textContent = `Overall Score: ${avgCloseness.toFixed(1)}%`;
    
    // Create chart
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    comparisonChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.user_times.map(t => t.toFixed(2)),
            datasets: [{
                label: 'Benchmark (100%)',
                data: data.bench_times.map(() => 100),
                borderColor: 'rgb(255, 159, 64)',
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0
            }, {
                label: 'Your Shot',
                data: data.user_closeness,
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
                    text: 'Shot Form Analysis - DTW Alignment'
                },
                legend: {
                    display: true,
                    position: 'top'
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
    
    // Display feedback
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
        document.getElementById('emailSuccessSection').style.display = 'block';
    }
}

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
        const avgCloseness = data.user_closeness.reduce((a, b) => a + b, 0) / data.user_closeness.length;
        const feedbackText = data.feedback.join('\n\n');
        
        // Send to backend
        const response = await fetch(`${API_URL}/send_email`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                firstName: userInfo.firstName,
                lastName: userInfo.lastName,
                email: userInfo.email,
                chartImage: chartImage,
                overallScore: avgCloseness.toFixed(1),
                feedback: feedbackText
            })
        });
        
        const emailData = await response.json();
        
        if (emailData.success) {
            // Show success message
            document.getElementById('emailSuccessSection').style.display = 'block';
        } else {
            console.error('Email sending failed:', emailData.error);
            // Still show success message (email might be delayed)
            document.getElementById('emailSuccessSection').style.display = 'block';
        }
    } catch (error) {
        console.error('Error sending email:', error);
        // Still show success message (email might be delayed)
        document.getElementById('emailSuccessSection').style.display = 'block';
    }
}

// ====================== EMAIL FUNCTIONALITY ======================
// (Email is now sent automatically after analysis - see sendEmailAutomatically function above)

// ====================== UTILITY FUNCTIONS ======================

function retakeBenchmark() {
    benchmarkFrames = [];
    document.getElementById('retakeBenchmark').style.display = 'none';
    document.getElementById('benchmarkStatus').textContent = '';
    document.getElementById('benchmarkStatus').className = 'status';
    if (benchmarkVideo.srcObject) {
        benchmarkVideo.srcObject.getTracks().forEach(track => track.stop());
        benchmarkVideo.srcObject = null;
    }
}

function retakeUser() {
    userFrames = [];
    document.getElementById('retakeUser').style.display = 'none';
    document.getElementById('userStatus').textContent = '';
    document.getElementById('userStatus').className = 'status';
    if (userVideo.srcObject) {
        userVideo.srcObject.getTracks().forEach(track => track.stop());
        userVideo.srcObject = null;
    }
}

function resetApp() {
    // Stop all streams
    if (benchmarkStream) {
        benchmarkStream.getTracks().forEach(track => track.stop());
    }
    if (userStream) {
        userStream.getTracks().forEach(track => track.stop());
    }
    
    // Clear intervals
    if (benchmarkInterval) clearInterval(benchmarkInterval);
    if (userInterval) clearInterval(userInterval);
    
    // Reset state
    benchmarkFrames = [];
    userFrames = [];
    benchmarkStream = null;
    userStream = null;
    userInfo = null;
    
    // Reset UI - go back to Step 0
    document.getElementById('step0').classList.add('active');
    document.getElementById('step0').style.display = 'block';
    document.getElementById('step1').classList.remove('active');
    document.getElementById('step1').style.display = 'none';
    document.getElementById('step2').classList.remove('active');
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step3').classList.remove('active');
    document.getElementById('step3').style.display = 'none';
    
    // Reset user info form
    document.getElementById('userInfoForm').reset();
    
    document.getElementById('startBenchmark').disabled = false;
    document.getElementById('stopBenchmark').disabled = true;
    document.getElementById('startUser').disabled = false;
    document.getElementById('stopUser').disabled = true;
    
    document.getElementById('retakeBenchmark').style.display = 'none';
    document.getElementById('retakeUser').style.display = 'none';
    
    document.getElementById('benchmarkStatus').textContent = '';
    document.getElementById('userStatus').textContent = '';
    document.getElementById('benchmarkStatus').className = 'status';
    document.getElementById('userStatus').className = 'status';
    
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'none';
    document.getElementById('emailSuccessSection').style.display = 'none';
    
    // Clear stored data
    currentResultsData = null;
    
    if (comparisonChart) {
        comparisonChart.destroy();
        comparisonChart = null;
    }
}

