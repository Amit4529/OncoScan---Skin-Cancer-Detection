// ---- API Configuration ----
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? `http://${window.location.host}`
    : 'https://amit0310-oncoscan.hf.space';

document.addEventListener('DOMContentLoaded', function () {
    // ---- DOM Elements ----
    const uploadContainer = document.getElementById('upload-container');
    const imageUpload = document.getElementById('image-upload');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const imagePreview = document.getElementById('image-preview');
    const scanForm = document.getElementById('scanForm');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultSection = document.getElementById('results-section');
    const resultContent = document.getElementById('results-content');
    const closeResults = document.getElementById('close-results');

    // ---- Model Metrics Modal ----
    const metricsModal = document.getElementById('metricsModal');
    const openMetricsBtn = document.getElementById('openMetricsBtn');
    const closeMetricsBtn = document.getElementById('closeMetricsBtn');

    if (openMetricsBtn && metricsModal) {
        openMetricsBtn.addEventListener('click', () => {
            metricsModal.classList.add('active');
            document.body.style.overflow = 'hidden';
        });
    }

    if (closeMetricsBtn && metricsModal) {
        closeMetricsBtn.addEventListener('click', () => {
            metricsModal.classList.remove('active');
            document.body.style.overflow = '';
        });

        metricsModal.addEventListener('click', (e) => {
            if (e.target === metricsModal) {
                metricsModal.classList.remove('active');
                document.body.style.overflow = '';
            }
        });
    }

    // ---- Navbar Scroll Effect ----
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 10) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // ---- Scroll Animations ----
    const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -40px 0px' };
    const scrollObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                scrollObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.feature-card, .about-card, .scan-form-card, .scan-info, .fact-highlight, .fact-desc').forEach(el => {
        el.style.opacity = '0';
        scrollObserver.observe(el);
    });

    // ---- Image Upload ----
    imageUpload.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                uploadPlaceholder.style.display = 'none';
                imagePreview.style.display = 'block';
                const img = document.createElement('img');
                img.src = e.target.result;
                imagePreview.innerHTML = '';
                imagePreview.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    });

    // ---- Drag & Drop ----
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, () => {
            uploadContainer.classList.add('highlight');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, () => {
            uploadContainer.classList.remove('highlight');
        }, false);
    });

    uploadContainer.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) {
            imageUpload.files = files;
            const event = new Event('change');
            imageUpload.dispatchEvent(event);
        }
    }, false);

    // ---- Form Submit → Gradio API ----
    scanForm.addEventListener('submit', function (e) {
        e.preventDefault();

        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;
        const image = imageUpload.files[0];

        if (!age || !gender || !image) {
            scanForm.style.animation = 'shakeX 0.5s';
            setTimeout(() => { scanForm.style.animation = ''; }, 600);
            return;
        }

        loadingOverlay.style.display = 'flex';

        const fileData = new FormData();
        fileData.append('files', image);

        // Gradio 6 native API: upload → call → SSE stream
        fetch(`${API_URL}/gradio_api/upload`, {
            method: 'POST',
            body: fileData
        })
            .then(response => response.json())
            .then(uploaded => {
                const fp = Array.isArray(uploaded) ? uploaded[0] : uploaded;
                const fileRef = (typeof fp === 'string')
                    ? { path: fp, meta: { _type: 'gradio.FileData' } }
                    : fp;
                return fetch(`${API_URL}/gradio_api/call/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data: [fileRef, Number(age), gender] })
                }).then(response => response.json());
            })
            .then(call => fetch(`${API_URL}/gradio_api/call/predict/${call.event_id}`))
            .then(response => response.text())
            .then(stream => {
                const lines = stream.split('\n').filter(l => l.startsWith('data: '));
                return JSON.parse(lines[lines.length - 1].slice(6))[0];
            })
            .then(data => {
                loadingOverlay.style.display = 'none';

                if (data.error) {
                    alert(`Error: ${data.error}`);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                loadingOverlay.style.display = 'none';
                alert(`Something went wrong: ${error.message}`);
            });
    });

    // ---- Display Results ----
    function displayResults(data) {
        const riskColors = {
            'HIGH': '#ef4444',
            'MODERATE': '#f59e0b',
            'LOW': '#22c55e'
        };
        const riskColor = riskColors[data.prediction.risk] || '#6b7280';
        const malProb = data.malignant_probability || 0;
        const isMal = malProb >= 50;

        // Top-3 predictions
        const top3Html = data.top3.map(p => `
            <div class="top3-item">
                <div class="top3-name">${p.name}</div>
                <div class="top3-bar-container">
                    <div class="top3-bar" style="width: ${p.probability}%; background: ${
                        p.severity === 'Malignant' || p.severity === 'Pre-cancerous' ? '#ef4444' : '#22c55e'
                    }"></div>
                </div>
                <div class="top3-prob">${p.probability}%</div>
            </div>
        `).join('');

        // Precautions
        const precautionsHtml = data.precautions.map(p =>
            `<li>${p}</li>`
        ).join('');

        const severityLabel = data.prediction.severity === 'Malignant' ? 'Cancerous'
            : data.prediction.severity === 'Benign' ? 'Non-Cancerous' : 'Pre-Cancerous';

        const html = `
            <div class="result-status" style="background: ${riskColor}10; border-left: 4px solid ${riskColor}; padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;">
                <div class="result-badge" style="background: ${riskColor}">
                    ${data.prediction.risk} RISK
                </div>
                <div style="font-size: 1.4em; font-weight: 700; color: var(--text); margin: 8px 0 4px;">
                    ${data.prediction.name}
                </div>
                <div style="font-size: 0.9em; color: ${riskColor}; font-weight: 500;">
                    ${severityLabel} &bull; ${data.prediction.confidence}% confidence
                </div>
            </div>

            <div class="result-item">
                <div class="result-label">About this condition</div>
                <div class="result-value">
                    ${data.prediction.description}
                </div>
            </div>

            <div class="result-item">
                <div class="result-label">Risk Assessment</div>
                <div class="malignancy-bar-container">
                    <div class="malignancy-bar" style="width: ${malProb}%; background: linear-gradient(90deg, ${isMal ? '#ef4444' : '#22c55e'}, ${isMal ? '#dc2626' : '#16a34a'});"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: var(--text-secondary);">
                    <span>Non-Cancerous</span>
                    <span style="font-weight: 600; color: ${isMal ? '#ef4444' : '#22c55e'};">
                        ${isMal ? 'Possibly Cancerous' : 'Likely Non-Cancerous'}
                    </span>
                    <span>Cancerous</span>
                </div>
            </div>

            <div class="result-item">
                <div class="result-label">Differential Diagnosis</div>
                ${top3Html}
            </div>

            <div class="recommendation">
                <div class="recommendation-title">
                    Recommended Next Steps
                </div>
                <ul>
                    ${precautionsHtml}
                </ul>
                <div class="disclaimer">
                    This is a screening tool only. It does not replace professional medical advice. Please consult a dermatologist for proper clinical diagnosis.
                </div>
            </div>
        `;

        resultContent.innerHTML = html;
        resultSection.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    // ---- Close Results ----
    closeResults.addEventListener('click', () => {
        resultSection.classList.remove('active');
        resultContent.innerHTML = '';
        document.body.style.overflow = '';
    });

    // Close results on overlay click
    resultSection.addEventListener('click', (e) => {
        if (e.target === resultSection) {
            resultSection.classList.remove('active');
            resultContent.innerHTML = '';
            document.body.style.overflow = '';
        }
    });
});

// ---- Shake animation (used for form validation) ----
const shakeStyle = document.createElement('style');
shakeStyle.textContent = `
    @keyframes shakeX {
        0%, 100% { transform: translateX(0); }
        20%, 60% { transform: translateX(-6px); }
        40%, 80% { transform: translateX(6px); }
    }
`;
document.head.appendChild(shakeStyle);
