// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {

const fileInput = document.getElementById('file');
const preview = document.getElementById('preview');
const previewWrapper = document.getElementById('preview-wrapper');
const resultDiv = document.getElementById('result');
const resultEmpty = document.getElementById('result-empty');
const errorDiv = document.getElementById('error');
const predClass = document.getElementById('pred-class');
const predConf = document.getElementById('pred-conf');
const probsBars = document.getElementById('probs-bars');
const dropzone = document.getElementById('dropzone');
const switchBtn = document.getElementById('switch-btn');
const switchModal = new bootstrap.Modal(document.getElementById('switchModelModal'));
const btnSwitchModel = document.getElementById('btnSwitchModel');
const resultsDirInput = document.getElementById('resultsDirInput');
const switchError = document.getElementById('switchError');
const darkModeBtn = document.getElementById('dark-mode-toggle');

// Initialize dark mode from localStorage
const initDarkMode = () => {
  const theme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-bs-theme', theme);
  updateIcon();
};

const toggleDarkMode = () => {
  const current = document.documentElement.getAttribute('data-bs-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-bs-theme', next);
  localStorage.setItem('theme', next);
  updateIcon();
};

const updateIcon = () => {
  const isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
  if (darkModeBtn) {
    darkModeBtn.innerHTML = isDark 
      ? '<i class="bi bi-sun-fill"></i>' 
      : '<i class="bi bi-moon-stars-fill"></i>';
  }
};

if (darkModeBtn) {
  darkModeBtn.addEventListener('click', toggleDarkMode);
}

// File upload
if (fileInput) {
  fileInput.addEventListener('change', handleFile);
}

function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  if (!file.type.startsWith('image/')) {
    alert('Please upload an image file.');
    return;
  }
  
  const reader = new FileReader();
  reader.onload = (ev) => {
    preview.src = ev.target.result;
    previewWrapper.style.display = 'block';
    predict(file);
  };
  reader.readAsDataURL(file);
}

// Drag & Drop
if (dropzone) {
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith('image/')) {
      alert('Please drop an image file.');
      return;
    }
    fileInput.files = e.dataTransfer.files;
    handleFile({ target: { files: [file] } });
  });
  dropzone.addEventListener('click', () => fileInput.click());
}

// Predict
async function predict(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  resultDiv.classList.add('d-none');
  resultEmpty.classList.remove('d-none');
  errorDiv.classList.add('d-none');
  
  try {
    const resp = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await resp.json();
    
    if (data.error) {
      showError(data.error);
      return;
    }
    
    showResult(data);
  } catch (err) {
    showError('Prediction failed. ' + err.message);
  }
}

function showResult(data) {
  // Top prediction
  predClass.textContent = data.pred_class;
  predConf.textContent = (data.confidence * 100).toFixed(1) + '%';
  
  // Probability bars
  const sorted = Object.entries(data.probs)
    .sort((a, b) => b[1] - a[1]);
  
  probsBars.innerHTML = sorted.map(([cls, prob]) => {
    const percent = (prob * 100).toFixed(1);
    return `
      <div class="prob-item">
        <div class="prob-label">
          <span class="prob-class">${cls}</span>
          <span class="prob-percent">${percent}%</span>
        </div>
        <div class="prob-bar-wrapper">
          <div class="prob-bar" style="width: ${percent}%"></div>
        </div>
      </div>
    `;
  }).join('');
  
  resultEmpty.classList.add('d-none');
  resultDiv.classList.remove('d-none');
}

function showError(msg) {
  errorDiv.textContent = msg;
  errorDiv.classList.remove('d-none');
  resultEmpty.classList.add('d-none');
}

// Model switching
if (switchBtn) {
  switchBtn.addEventListener('click', () => switchModal.show());
}

if (btnSwitchModel) {
  btnSwitchModel.addEventListener('click', async () => {
    const dir = resultsDirInput.value.trim();
    if (!dir) {
      alert('Please enter results directory path.');
      return;
    }
    
    btnSwitchModel.disabled = true;
    btnSwitchModel.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Switching...';
    switchError.classList.add('d-none');
    
    try {
      const resp = await fetch('/api/switch_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ results_dir: dir })
      });
      const data = await resp.json();
      
      if (resp.ok) {
        alert(`Switched to ${dir}!`);
        switchModal.hide();
        location.reload();
      } else {
        switchError.textContent = data.error || 'Failed to switch model.';
        switchError.classList.remove('d-none');
      }
    } catch (err) {
      switchError.textContent = 'Error: ' + err.message;
      switchError.classList.remove('d-none');
    } finally {
      btnSwitchModel.disabled = false;
      btnSwitchModel.innerHTML = '<i class="bi bi-check-lg me-1"></i>Apply Changes';
    }
  });
}

// Init
initDarkMode();

}); // End DOMContentLoaded
