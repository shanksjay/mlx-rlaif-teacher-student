document.addEventListener('DOMContentLoaded', function() {
  const statusDiv = document.getElementById('status');
  const checkBtn = document.getElementById('check-system');
  const outputDiv = document.getElementById('output');

  // Simple check to local server
  fetch('http://localhost:8000/health')
    .then(res => res.json())
    .then(data => {
      statusDiv.textContent = 'System Online';
      statusDiv.className = 'status online';
    })
    .catch(err => {
      statusDiv.textContent = 'System Offline';
      statusDiv.className = 'status offline';
    });

  checkBtn.addEventListener('click', () => {
    fetch('http://localhost:8000/system_specs')
      .then(res => res.json())
      .then(data => {
        outputDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
      })
      .catch(err => {
        outputDiv.textContent = 'Error fetching specs: ' + err;
      });
  });
});
