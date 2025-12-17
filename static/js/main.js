const form = document.getElementById('upload-form');
const status = document.getElementById('status');
const results = document.getElementById('results');

form.addEventListener('submit', async (e) => {
	e.preventDefault();
	status.innerText = 'Uploading and processing...';
	results.style.display = 'none';

	const fileInput = document.getElementById('file');
	if (!fileInput.files.length) {
		status.innerText = 'Please select a CSV file.';
		return;
	}

	const formData = new FormData();
	formData.append('file', fileInput.files[0]);
	formData.append('auto_clean', document.getElementById('auto_clean').value);
	formData.append('impute_method', document.getElementById('impute_method').value);

	try {
		const resp = await fetch('/upload', {
			method: 'POST',
			body: formData
		});

		const data = await resp.json();

		if (!resp.ok) {
			status.innerText = 'Error: ' + (data.error || 'Unknown error');
			return;
		}

		status.innerText = 'Processing complete!';
		results.style.display = 'block';

        // Display summary
		document.getElementById('summary').innerText = data.summary;

        // Display generated plots
		const plotsDiv = document.getElementById('plots');
		plotsDiv.innerHTML = "";  
		data.plots.forEach((plot) => {
			const img = document.createElement('img');
			img.src = `/static/outputs/${plot}`;
			img.className = 'plot-image';
			plotsDiv.appendChild(img);
		});

	} catch (err) {
		console.error(err);
		status.innerText = 'An unexpected error occurred.';
	}
});