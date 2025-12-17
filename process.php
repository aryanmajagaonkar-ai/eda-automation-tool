<?php

echo "
<style>
body{
    font-family: 'Inter', sans-serif;
    background: #0f172a;
    color: #fff;
    padding: 40px;
}
.card{
    background: #1e293b;
    padding: 20px;
    margin: 20px 0;
    border-radius: 16px;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
}
.stats{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}
.stat-box{
    background:#020617;
    padding: 20px;
    border-radius: 14px;
    min-width: 200px;
    text-align:center;
}
.stat-box h2{
    font-size:30px;
    margin:0;
    color: #4dd0e1;
}
.stat-box p{
    margin:5px 0 0 0;
    font-size:14px;
    opacity:0.8;
}
img{
    border-radius: 14px;
    margin: 20px 0;
    width: 100%;
    max-width: 800px;
}
</style>
";

// Disable warnings for clean output
error_reporting(E_ALL);

// Folders
$uploadDir = __DIR__ . "/uploads/";
$outputDir = __DIR__ . "/output/";

// Create folders if missing
if (!file_exists($uploadDir)) mkdir($uploadDir, 0777, true);
if (!file_exists($outputDir)) mkdir($outputDir, 0777, true);

// 1. Check file received
if (!isset($_FILES['dataset']) || $_FILES['dataset']['error'] !== UPLOAD_ERR_OK) {
    die("ERROR: No CSV file uploaded!");
}

$file = $_FILES['dataset'];
$filename = basename($file['name']);
$filepath = $uploadDir . $filename;

// 2. Move uploaded CSV
if (!move_uploaded_file($file['tmp_name'], $filepath)) {
    die("ERROR: Failed to save uploaded file.");
}

echo "SUCCESS: File uploaded at $filepath <br><br>";

$filepath = $uploadDir . $filename;

// === START OF CHANGE ===
// Collect the answers from the form
$a1 = $_POST['ans_missing']; // e.g. "2"
$a2 = $_POST['ans_drop'];    // e.g. "y"
$a3 = $_POST['ans_menu'];    // e.g. "1"

// Combine them with commas. Order MUST match your Python input() calls!
// Order: Missing Value -> Drop Cols -> Main Menu
$inputString = "$a1,$a2,$a3"; 
// === END OF CHANGE ===

// 3. Run Python EDA pipeline
$ch = curl_init();

curl_setopt($ch, CURLOPT_URL, "http://localhost:10000/run-eda");
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

$data = [
    "file" => new CURLFile($filepath),
    "user_inputs" => $inputString // <--- Add this line to send the string to Flask
];

curl_setopt($ch, CURLOPT_POSTFIELDS, $data);

$response = curl_exec($ch);

if(curl_errno($ch)){
    die("CURL ERROR: " . curl_error($ch));
}

curl_close($ch);

echo "<h2>Flask Response</h2>";
echo "<pre>$response</pre>";

// 4. Load summary.json
$summaryPath = $outputDir . "/summary.json";

if (!file_exists($summaryPath)) {
    die("<h3>ERROR: summary.json not found. Python script failed.</h3>");
}

$summaryData = json_decode(file_get_contents($summaryPath), true);

// Display summary
echo "<div class='card'><h1>📊 Dataset Summary</h1>";
echo "<div class='stats'>";

echo "<div class='stat-box'><h2>{$summaryData['rows']}</h2><p>Rows</p></div>";
echo "<div class='stat-box'><h2>{$summaryData['columns']}</h2><p>Columns</p></div>";
echo "<div class='stat-box'><h2>{$summaryData['missing_cells']}</h2><p>Missing Values</p></div>";
echo "<div class='stat-box'><h2>{$summaryData['duplicate_rows']}</h2><p>Duplicate Rows</p></div>";

echo "</div></div>";

// 5. Display generated images
echo "<h2>Generated Visualizations</h2>";

// Use the local path to find images (assuming 'output' is in the same directory as this PHP script)
$images = glob("output/*.png"); 
// Note: Changed $outputDir . "/*.png" to "output/*.png" for simpler relative path display

if (empty($images)) {
    echo "<p>⚠️ No images found in the 'output' folder. Check Python script logs or Flask console for errors.</p>";
}

foreach ($images as $img) {
    $imgName = basename($img);
    // Use the relative path from the PHP script location (e.g., /output/correlation.png)
    // This avoids reliance on the Flask server for image serving if PHP/Apache can access it directly.
    echo "<img src='./output/$imgName' alt='$imgName' style='margin:15px;border-radius:10px;'>";
}

echo "<br><a href='index.php'>Run Another File</a>";
?>
