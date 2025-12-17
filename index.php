<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Automation Tool</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.birds.min.js"></script>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        #vanta-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        
        body { 
            padding: 0;
            position: relative;
            min-height: 100vh;

            display: flex;
            justify-content: center;   /* Horizontal center */
            align-items: center;       /* Vertical center */
        }
        
        .container {
            background: rgba(0, 0, 0, 0.6);
            color: #fefefeff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
            max-width: 600px;
        }
        
        h1 {
            text-align: center;
            color: #a1a1adff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .step-container { 
            display: none;
            animation: fadeIn 0.5s ease-in;
        }
        
        .step-container.active { 
            display: block; 
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .output-box { 
            background:  #99cfcfff; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 15px 0; 
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
        }
        
        .img-output { 
            max-width: 100%; 
            height: auto; 
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .loading { 
            display: none; 
            text-align: center; 
            padding: 20px;
            background: rgba(114, 175, 190, 0.9);
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .loading.active { 
            display: block; 
        }
        
        .card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .workflow-card h4 {
            font-size: 1.4rem;
            margin-bottom: 8px;
            font-weight: 600;
            color: #ffffff;
        }

        .workflow-card p {
            font-size: 0.95rem;
            color: #d3d3d3;
            margin-bottom: 18px;
        }

        .workflow-card {
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 22px;
            margin: 10px 0;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);

            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
            transition: transform 0.25s ease,
                        box-shadow 0.25s ease,
                        background 0.25s ease;

        }

        .workflow-card:hover {
            transform: translateY(-6px);
            background: rgba(255, 255, 255, 0.12);
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.5);
            border-color: rgba(255, 255, 255, 0.25);
        }

        .workflow-card button {
            background: #0066ff;
            border: none;
            padding: 10px 18px;
            border-radius: 10px;
            color: #fff;
            font-weight: 600;
            transition: background 0.2s ease, transform 0.2s ease;
        }

        .workflow-card button:hover {
            background: #0053d6;
            transform: translateY(-2px);
        }

       
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        
        .btn {
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: scale(1.05);
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 1000;
        }

        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-dot.connected { background: #28a745; }
        .status-dot.disconnected { background: #dc3545; }
        .status-dot.checking { background: #ffc107; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .alert-custom {
            animation: slideDown 0.3s ease-out;
        }
        
        @keyframes slideDown {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <!-- Vanta.js Background -->
    <div id="vanta-bg"></div>
    
    <!-- Connection Status Indicator -->
    <div class="connection-status">
        <span class="status-dot checking" id="statusDot"></span>
        <span id="statusText">Checking connection...</span>
    </div>
    
    <div class="container">
        <h1 class="mb-4">🤖 AI-Powered EDA Tool</h1>
        
        <!-- Connection Error Alert -->
        <div id="connectionAlert" class="alert alert-danger alert-custom" style="display:none;">
            <strong>⚠️ Connection Error!</strong>
            <p>Cannot connect to Flask backend. Please ensure:</p>
            <ul>
                <li>Flask server is running: <code>python app.py</code></li>
                <li>Server is running on: <code>http://localhost:5000</code></li>
                <li>No firewall blocking the connection</li>
            </ul>
            <button class="btn btn-sm btn-warning" onclick="checkConnection()">Retry Connection</button>
        </div>
        
        <!-- Loading Spinner -->
        <div class="loading" id="loading">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Processing your data...</p>
        </div>
        
        <!-- Step 1: Upload CSV -->
        <div class="step-container active" id="step1">
            <h3>📁 Step 1: Upload CSV File</h3>
            <div class="mb-3">
                <input type="file" class="form-control" id="csvFile" accept=".csv">
                <small class="text-muted">Supported format: .csv (Max 16MB)</small>
            </div>
            <button class="btn btn-primary btn-lg" onclick="uploadCSV()">
                <i class="bi bi-upload"></i> Upload & Analyze
            </button>
        </div>
        
        <!-- Step 2: AI Suggestions -->
        <div class="step-container" id="step2">
            <h3>🔍 Step 2: Dataset Overview & AI Suggestions</h3>
            <div id="datasetInfo" class="output-box"></div>
            <div id="suggestions" class="output-box"></div>
            <img id="heatmap1" class="img-output" style="display:none;">
            
            <h4 class="mt-3">Choose Action:</h4>
            <div class="btn-group" role="group">
                <button class="btn btn-success" onclick="handleSuggestions('auto')">
                    ✨ Auto Clean (AI Recommended)
                </button>
                <button class="btn btn-warning" onclick="showCustomDropForm()">
                    🎯 Drop Custom Columns
                </button>
                <button class="btn btn-secondary" onclick="handleSuggestions('skip')">
                    ⏭️ Skip & Continue
                </button>
            </div>
            
            <div id="customDropForm" style="display:none;" class="mt-3">
                <label>Enter columns to drop (comma-separated):</label>
                <input type="text" class="form-control" id="customColumns" placeholder="col1, col2, col3">
                <button class="btn btn-primary mt-2" onclick="handleSuggestions('manual')">Drop Selected</button>
            </div>
        </div>
        
        <!-- Step 3: Missing Values -->
        <div class="step-container" id="step3">
            <h3>🔧 Step 3: Handle Missing Values</h3>
            <div id="missingInfo" class="output-box"></div>
            <img id="heatmap2" class="img-output" style="display:none;">
            
            <h4>Choose Imputation Method:</h4>
            <div class="btn-group" role="group">
                <button class="btn btn-primary" onclick="handleMissing('mean')">📊 Mean</button>
                <button class="btn btn-primary" onclick="handleMissing('median')">📈 Median</button>
                <button class="btn btn-primary" onclick="handleMissing('mode')">🎯 Mode</button>
            </div>
        </div>
        
        <!-- Step 4: Choose Workflow -->
        <div class="step-container" id="step4">
            <h3 class="step-title">🎯 Step 4: Choose Workflow</h3>
            <div class="row mt-3">
                <div class="col-md-4">
                    <div class="card">
                        <div class="workflow-card">
                            <h5 class="card-title">📊 EDA</h5>
                            <p>Exploratory Data Analysis & Insights</p>
                            <button class="btn btn-primary" onclick="chooseWorkflow('eda')">Select EDA</button>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="workflow-card">
                            <h5 class="card-title">🤖 Machine Learning</h5>
                            <p>Prepare data for ML models</p>
                            <button class="btn btn-primary" onclick="chooseWorkflow('ml')">Select ML</button>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="workflow-card">
                            <h5 class="card-title">📈 Dashboard</h5>
                            <p>Create visualizations</p>
                            <button class="btn btn-primary" onclick="chooseWorkflow('dashboard')">Select Dashboard</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Step 5: EDA Results -->
        <div class="step-container" id="step5_eda">
            <h3>📊 EDA Results</h3>
            <div id="edaResults" class="output-box"></div>
            <button class="btn btn-success" onclick="downloadFile('eda')">💾 Download EDA Data</button>
            <button class="btn btn-secondary" onclick="resetTool()">🔄 Start Over</button>
        </div>
        
        <!-- Step 5: ML Preparation -->
        <div class="step-container" id="step5_ml">
            <h3>🤖 Machine Learning Preparation</h3>
            <div id="mlColumns" class="output-box"></div>
            
            <div class="mb-3">
                <label>Select Target Column:</label>
                <select class="form-control" id="targetColumn"></select>
            </div>

            <div class="mb-3">
                <label>Choose Scaling Method:</label>
                <select class="form-control" id="scalingMethod">
                    <option value="standard">Standard Scaling</option>
                    <option value="minmax">Min-Max Scaling</option>
                    <option value="robust">Robust Scaling</option>
                    <option value="gaussian">Gaussian Transformation</option>
                </select>
            </div>

            <div id="qqPlotContainer" style="display:none; margin-top:20px;">
                <h5>Q-Q Plot</h5>
                <img id="qqPlotImage" src="" class="img-fluid" />
            </div>

            <div id="gaussianOptions" class="mb-3" style="display:none;">
                <label>Select Gaussian Method:</label>
                <select class="form-control" id="gaussianMethod">
                    <option value="log">Log Transformation</option>
                    <option value="sqrt">Square Root</option>
                    <option value="reciprocal">Reciprocal</option>
                    <option value="boxcox">Box-Cox Transformation</option>
                    <option value="exp">Exponential</option>
                </select>
            </div>
            
            <button class="btn btn-primary" onclick="prepareML()">🚀 Prepare ML Data</button>
            
            <div id="mlResults" class="output-box" style="display:none;"></div>
            <div id="mlButtons" style="display:none;">
                <button class="btn btn-success" onclick="downloadFile('train')">💾 Download Train Data</button>
                <button class="btn btn-success" onclick="downloadFile('test')">💾 Download Test Data</button>
                <button class="btn btn-secondary" onclick="resetTool()">🔄 Start Over</button>
            </div>
        </div>
        
        <!-- Step 5: Dashboard -->
        <div class="step-container" id="step5_dashboard">
            <h3>📈 Create Visualization</h3>
            
            <div class="row">
                <div class="col-md-6">
                    <label>Visualization Type:</label>
                    <select class="form-control" id="vizType" onchange="updateVizInputs()">
                        <option value="bar">Bar Chart</option>
                        <option value="line">Line Chart</option>
                        <option value="scatter">Scatter Plot</option>
                        <option value="pie">Pie Chart</option>
                        <option value="hist">Histogram</option>
                        <option value="heatmap">Correlation Heatmap</option>
                    </select>
                </div>
            </div>
            
            <div class="row mt-3" id="vizInputs">
                <div class="col-md-6">
                    <label>X-Axis Column:</label>
                    <select class="form-control" id="xColumn"></select>
                </div>
                <div class="col-md-6">
                    <label>Y-Axis Column:</label>
                    <select class="form-control" id="yColumn"></select>
                </div>
            </div>
            
            <button class="btn btn-primary mt-3" onclick="createVisualization()">🎨 Generate Chart</button>
            
            <div id="vizOutput" class="mt-3"></div>
            <button class="btn btn-secondary mt-3" onclick="resetTool()">🔄 Start Over</button>
        </div>
    </div>

    <script>
        // Initialize Vanta.js Birds Background
        let vantaEffect;
        window.addEventListener('DOMContentLoaded', () => {
            vantaEffect = VANTA.BIRDS({
                el: "#vanta-bg",
                mouseControls: true,
                touchControls: true,
                gyroControls: false,
                minHeight: 200.00,
                minWidth: 200.00,
                scale: 1.00,
                scaleMobile: 1.00,
                backgroundColor: 0x1a1a1e,
                color1: 0x4dd0e1,
                color2: 0xffffff,
                colorMode: "lerp",
                birdSize: 1.20,
                wingSpan: 30.00,
                speedLimit: 4.00,
                separation: 30.00,
                alignment: 20.00,
                cohesion: 30.00,
                quantity: 3.00
            });
        });

        const FLASK_URL = 'https://eda-automation-tool.onrender.com';
        const sessionId = Date.now().toString();
        let currentData = {};
        let isConnected = false;
        
        // Check Flask connection on page load
        window.addEventListener('load', () => {
            checkConnection();
        });
        
        async function checkConnection() {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const connectionAlert = document.getElementById('connectionAlert');
            
            statusDot.className = 'status-dot checking';
            statusText.textContent = 'Checking connection...';
            
            try {
                const response = await fetch(`${FLASK_URL}/health`, {
                    method: 'GET',
                    signal: AbortSignal.timeout(5000) // 5 second timeout
                });
                
                if (response.ok) {
                    isConnected = true;
                    statusDot.className = 'status-dot connected';
                    statusText.textContent = 'Connected';
                    connectionAlert.style.display = 'none';
                } else {
                    throw new Error('Server responded with error');
                }
            } catch (error) {
                isConnected = false;
                statusDot.className = 'status-dot disconnected';
                statusText.textContent = 'Disconnected';
                connectionAlert.style.display = 'block';
                console.error('Connection check failed:', error);
            }
        }
        
        function showLoading(show) {
            document.getElementById('loading').classList.toggle('active', show);
        }
        
        function showStep(stepId) {
            document.querySelectorAll('.step-container').forEach(el => el.classList.remove('active'));
            document.getElementById(stepId).classList.add('active');
        }
        
        async function uploadCSV() {
            const fileInput = document.getElementById('csvFile');
            if (!fileInput.files[0]) {
                alert('Please select a CSV file');
                return;
            }
            
            // Check file size (16MB limit)
            if (fileInput.files[0].size > 16 * 1024 * 1024) {
                alert('File too large! Maximum size is 16MB');
                return;
            }
            
            // Check connection first
            if (!isConnected) {
                alert('⚠️ Not connected to Flask server. Please start the server and retry.');
                await checkConnection();
                return;
            }
            
            showLoading(true);
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('session_id', sessionId);
            
            try {
                const response = await fetch(`${FLASK_URL}/upload_csv`, {
                    method: 'POST',
                    body: formData,
                    signal: AbortSignal.timeout(30000) // 30 second timeout
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    currentData = data;
                    displayDatasetInfo(data);
                    await getAISuggestions();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Upload error:', error);
                if (error.name === 'TimeoutError') {
                    alert('⏱️ Request timed out. The file might be too large or server is slow.');
                } else if (error.message.includes('Failed to fetch')) {
                    alert('❌ Connection error! Please ensure Flask server is running on http://localhost:5000');
                    await checkConnection();
                } else {
                    alert('Error: ' + error.message);
                }
            } finally {
                showLoading(false);
            }
        }
        
        function displayDatasetInfo(data) {
            document.getElementById('datasetInfo').innerHTML = `
                <h5>Dataset Information:</h5>
                <p><strong>Shape:</strong> ${data.shape[0]} rows × ${data.shape[1]} columns</p>
                <p><strong>Numeric Columns:</strong> ${data.numeric_cols.length}</p>
                <p><strong>Categorical Columns:</strong> ${data.categorical_cols.length}</p>
                <p><strong>Missing Values:</strong> ${data.missing_values}</p>

                <h5 class="mt-3">Missing Values Per Column:</h5>
                <table class="table table-bordered table-sm">
                   <thead>
                      <tr>
                          <th>Column</th>
                          <th>Missing Count</th>
                      </tr>
                  </thead>
                  <tbody>
                      ${Object.entries(data.missing_per_column)
                          .map(([col, val]) => `
                              <tr style="background-color: ${val > 0 ? '#fff3cd' : 'white'};">
                                  <td>${col}</td>
                                  <td><strong style="color:${val > 0 ? 'red' : 'black'};">${val}</strong></td>
                              </tr>
                           `)
                           .join('')}
                  </tbody>
               </table>


                <div class="mt-3">${data.head}</div>
            `;
            
            if (data.missing_heatmap) {
                const heatmap = document.getElementById('heatmap1');
                heatmap.src = 'data:image/png;base64,' + data.missing_heatmap;
                heatmap.style.display = 'block';
            }
        }
        
        async function getAISuggestions() {
            try {
                const response = await fetch(`${FLASK_URL}/ai_suggestions`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId}),
                    signal: AbortSignal.timeout(15000)
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    let html = '<h5>🤖 AI Suggestions:</h5><ul>';
                    data.suggestions.forEach(s => {
                        html += `<li><span class="badge bg-${s.severity === 'high' ? 'danger' : 'warning'}">${s.severity}</span> ${s.message}</li>`;
                    });
                    html += '</ul>';
                    
                    if (data.suggested_drop_columns.length > 0) {
                        html += `<p><strong>Suggested columns to drop:</strong> ${data.suggested_drop_columns.join(', ')}</p>`;
                    }
                    
                    document.getElementById('suggestions').innerHTML = html;
                    showStep('step2');
                }
            } catch (error) {
                console.error('AI suggestions error:', error);
                alert('Error getting AI suggestions: ' + error.message);
            }
        }
        
        function showCustomDropForm() {
            document.getElementById('customDropForm').style.display = 'block';
        }
        
        async function handleSuggestions(action) {
            showLoading(true);
            
            let payload = {
                session_id: sessionId,
                action: action
            };
            
            if (action === 'manual') {
                const customCols = document.getElementById('customColumns').value
                    .split(',').map(s => s.trim()).filter(s => s);
                payload.columns = customCols;
            }
            
            try {
                const response = await fetch(`${FLASK_URL}/handle_suggestions`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload),
                    signal: AbortSignal.timeout(15000)
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    if (data.missing_values > 0) {
                        document.getElementById('missingInfo').innerHTML = `
                            <p class="alert alert-warning">⚠️ ${data.missing_values} missing values detected</p>
                        `;
                        if (data.missing_heatmap) {
                            const heatmap = document.getElementById('heatmap2');
                            heatmap.src = 'data:image/png;base64,' + data.missing_heatmap;
                            heatmap.style.display = 'block';
                        }
                        showStep('step3');
                    } else {
                        showStep('step4');
                    }
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                showLoading(false);
            }
        }
        
        async function handleMissing(method) {
            showLoading(true);
            
            try {
                const response = await fetch(`${FLASK_URL}/missing_values`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        method: method
                    }),
                    signal: AbortSignal.timeout(15000)
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    alert(data.message);
                    showStep('step4');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                showLoading(false);
            }
        }
        
        async function chooseWorkflow(workflow) {
            showLoading(true);
            
            try {
                const response = await fetch(`${FLASK_URL}/workflow`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        workflow: workflow
                    }),
                    signal: AbortSignal.timeout(15000)
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    if (workflow === 'eda') {
                        document.getElementById('edaResults').innerHTML = `
                            <h5>✅ EDA Complete!</h5>
                            <p><strong>Total Rows:</strong> ${data.summary.total_rows}</p>
                            <p><strong>Total Columns:</strong> ${data.summary.total_columns}</p>
                            <p><strong>Numeric Columns:</strong> ${data.summary.numeric_columns}</p>
                            <p><strong>Categorical Columns:</strong> ${data.summary.categorical_columns}</p>
                            <p><strong>Missing Cells:</strong> ${data.summary.missing_cells}</p>
                        `;
                        showStep('step5_eda');
                    } else if (workflow === 'ml') {

                        // Populate Target Dropdown
                        const select = document.getElementById('targetColumn');
                        select.innerHTML = '';
                        data.columns.forEach(col => {
                            const option = document.createElement('option');
                            option.value = col;
                            option.text = col;
                            select.appendChild(option);
                        });

                        // Show ML Step
                        showStep('step5_ml');

                    

                    
                    } else if (workflow === 'dashboard') {
                        populateVizColumns(data.numeric_columns, data.categorical_columns);
                        showStep('step5_dashboard');
                    }
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                showLoading(false);
            }
        }

        async function generateQQPlot() {
           const target = document.getElementById("targetColumn").value;

           if (!target) {
               console.warn("Target column not selected yet.");
               return;
            }

           showLoading(true);

           try {
                const response = await fetch(`${FLASK_URL}/generate_qqplot`, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        session_id: sessionId,
                        target_column: target
                    }),
                    signal: AbortSignal.timeout(15000)
                });

                const data = await response.json();

                if (data.status === "success") {
                    document.getElementById("qqPlotImage").src =
                        "data:image/png;base64," + data.qqplot;
                    document.getElementById("qqPlotContainer").style.display = "block";
                } else {
                    console.error("Q-Q plot generation failed:", data.error);
                }
            } catch (err) {
                console.error("Q-Q plot error:", err);
            } finally {
                showLoading(false);
            }
        }

        
        document.getElementById("scalingMethod").addEventListener("change", function () {
            if (this.value === "gaussian") {
                document.getElementById("gaussianOptions").style.display = "block";
                generateQQPlot();  // call backend
            } else {
                document.getElementById("gaussianOptions").style.display = "none";
                document.getElementById("qqPlotContainer").style.display = "none";
            }
        });



        // ----------------- FIXED ML PREP FUNCTION -----------------
        async function prepareML() {
           showLoading(true);

           try {
               let scaling = document.getElementById("scalingMethod").value;
               let gaussianMethod = document.getElementById("gaussianMethod")?.value || null;

               // Payload
               let payload = {
                   session_id: sessionId,
                   target_column: document.getElementById("targetColumn").value,
                   transform: scaling,
                   gaussian_method: gaussianMethod
                };

                // -------- FLASK ML PREP CALL --------
                const response = await fetch(`${FLASK_URL}/ml_prepare`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                    signal: AbortSignal.timeout(20000)
                });

                const data = await response.json();

                if (data.status === "success") {

                    document.getElementById("mlResults").innerHTML = `
                        <h5>✅ ML Preparation Complete!</h5>
                        <p><strong>Problem Type:</strong> ${data.problem_type}</p>
                        <p><strong>Train Shape:</strong> ${data.train_shape[0]} × ${data.train_shape[1]}</p>
                        <p><strong>Test Shape:</strong> ${data.test_shape[0]} × ${data.test_shape[1]}</p>

                        <h6 class="mt-2">Transformations Applied:</h6>
                        <ul>
                            ${data.transformations_applied.map(t => `<li>${t}</li>`).join("")}
                        </ul>
                    `;

                    document.getElementById("mlResults").style.display = "block";
                    document.getElementById("mlButtons").style.display = "block";

                } else {
                    alert("❌ ML Preparation Failed: " + data.error);
                }

            } catch (err) {
                console.error(err);
                alert("⚠ Error preparing ML: " + err.message);
            } finally {
                showLoading(false);
            }
        }

        
        function populateVizColumns(numCols, catCols) {
            const xSelect = document.getElementById('xColumn');
            const ySelect = document.getElementById('yColumn');
            
            xSelect.innerHTML = '';
            ySelect.innerHTML = '';
            
            [...numCols, ...catCols].forEach(col => {
                xSelect.innerHTML += `<option value="${col}">${col}</option>`;
                ySelect.innerHTML += `<option value="${col}">${col}</option>`;
            });
        }
        
        function updateVizInputs() {
            const vizType = document.getElementById('vizType').value;
            const inputs = document.getElementById('vizInputs');
            
            if (vizType === 'heatmap') {
                inputs.style.display = 'none';
            } else if (vizType === 'pie' || vizType === 'hist') {
                inputs.style.display = 'block';
                document.querySelector('#vizInputs .col-md-6:last-child').style.display = 'none';
            } else {
                inputs.style.display = 'block';
                document.querySelector('#vizInputs .col-md-6:last-child').style.display = 'block';
            }
        }
        
        async function createVisualization() {
            showLoading(true);
            
            try {
                const response = await fetch(`${FLASK_URL}/create_visualization`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        viz_type: document.getElementById('vizType').value,
                        x_column: document.getElementById('xColumn').value,
                        y_column: document.getElementById('yColumn').value
                    }),
                    signal: AbortSignal.timeout(20000)
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    document.getElementById('vizOutput').innerHTML = `
                        <img src="data:image/png;base64,${data.visualization}" class="img-fluid">
                    `;
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                showLoading(false);
            }
        }
        
        async function downloadFile(fileType) {
            try {
                const response = await fetch(`${FLASK_URL}/download`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        file_type: fileType
                    }),
                    signal: AbortSignal.timeout(15000)
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    const blob = new Blob([data.data], {type: 'text/csv'});
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = data.filename;
                    a.click();
                }
            } catch (error) {
                alert('Download error: ' + error.message);
            }
        }
        
        function resetTool() {
            location.reload();
        }
    </script>
</body>
</html>