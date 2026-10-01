document.addEventListener('DOMContentLoaded', function () {
    const uploadContainer = document.getElementById('upload-container');
    const imageUpload = document.getElementById('image-upload');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const imagePreview = document.getElementById('image-preview');
    const scanForm = document.getElementById('scanForm');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultSection = document.getElementById('results-section');
    const resultContent = document.getElementById('results-content');
    const closeResults = document.getElementById('close-results');

    // Animate counters
    const counters = document.querySelectorAll('.counter');
    const speed = 200;
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && entry.target.classList.contains('counter')) {
                const target = parseInt(entry.target.getAttribute('data-target'));
                let count = 0;
                const updateCount = () => {
                    const increment = target / speed;
                    if (count < target) {
                        count += increment;
                        entry.target.innerText = Math.ceil(count);
                        setTimeout(updateCount, 1);
                    } else {
                        entry.target.innerText = target;
                    }
                };
                updateCount();
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    counters.forEach(counter => observer.observe(counter));

    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.info-card, .scan-form');
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            if (elementPosition < screenPosition) {
                element.classList.add('animate__animated', 'animate__fadeInUp');
            }
        });
    };
    window.addEventListener('scroll', animateOnScroll);

    imageUpload.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                uploadPlaceholder.style.display = 'none';
                imagePreview.style.display = 'block';
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('animate__animated', 'animate__fadeIn');
                imagePreview.innerHTML = '';
                imagePreview.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadContainer.classList.add('highlight');
        uploadContainer.style.borderColor = 'var(--primary-color)';
        uploadContainer.style.backgroundColor = 'rgba(0, 102, 204, 0.05)';
        uploadContainer.style.transform = 'scale(1.02)';
    }

    function unhighlight() {
        uploadContainer.classList.remove('highlight');
        uploadContainer.style.borderColor = 'var(--border-color)';
        uploadContainer.style.backgroundColor = 'rgba(255, 255, 255, 0.5)';
        uploadContainer.style.transform = 'scale(1)';
    }

    uploadContainer.addEventListener('drop', handleDrop, false);
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            imageUpload.files = files;
            const event = new Event('change');
            imageUpload.dispatchEvent(event);
        }
    }

    scanForm.addEventListener('submit', function (e) {
        e.preventDefault();

        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;
        const bodySite = document.getElementById('body-site').value;
        const image = imageUpload.files[0];

        if (!age || !gender || !image) {
            scanForm.classList.add('animate__animated', 'animate__shakeX');
            setTimeout(() => {
                scanForm.classList.remove('animate__animated', 'animate__shakeX');
            }, 1000);
            return;
        }

        loadingOverlay.style.display = 'flex';
        loadingOverlay.classList.add('animate__animated', 'animate__fadeIn');

        const formData = new FormData();
        formData.append('age', age);
        formData.append('gender', gender);
        formData.append('body_site', bodySite);
        formData.append('image', image);

        // fetch('https://oncoscan-skin-cancer-detection.onrender.com/predict', {
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                loadingOverlay.classList.remove('animate__fadeIn');
                loadingOverlay.classList.add('animate__fadeOut');

                setTimeout(() => {
                    loadingOverlay.style.display = 'none';
                    loadingOverlay.classList.remove('animate__fadeOut');

                    if (data.error) {
                        alert(`Error: ${data.error}`);
                    } else {
                        const riskColors = {
                            'HIGH': '#ef4444',
                            'MODERATE': '#f59e0b',
                            'LOW': '#22c55e'
                        };
                        const riskColor = riskColors[data.prediction.risk] || '#6b7280';
                        const malProb = data.malignant_probability || 0;

                        // Top-3 predictions HTML
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

                        // Precautions HTML
                        const precautionsHtml = data.precautions.map(p =>
                            `<li>${p}</li>`
                        ).join('');

                        const html = `
                            <div class="result-item">
                                <div class="result-badge" style="background: ${riskColor}">
                                    ${data.prediction.risk} RISK
                                </div>
                                <div class="result-label">Detected Condition</div>
                                <div class="result-value" style="font-size: 1.3em; font-weight: 600;">
                                    ${data.prediction.name}
                                </div>
                                <div class="result-label" style="margin-top: 4px; opacity: 0.7;">
                                    ${data.prediction.severity} • ${data.prediction.confidence}% confidence
                                </div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Description</div>
                                <div class="result-value" style="font-size: 0.95em;">
                                    ${data.prediction.description}
                                </div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Malignancy Assessment</div>
                                <div class="malignancy-bar-container">
                                    <div class="malignancy-bar" style="width: ${malProb}%; background: ${
                                        malProb >= 50 ? '#ef4444' : '#22c55e'
                                    }"></div>
                                </div>
                                <div class="result-value" style="font-size: 0.9em;">
                                    ${malProb >= 50 ? '⚠️ Potentially Malignant' : '✅ Likely Benign'}
                                    (${malProb}% malignant probability)
                                </div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Top Predictions</div>
                                ${top3Html}
                            </div>
                            <div class="recommendation">
                                <div class="recommendation-title">Recommended Actions</div>
                                <ul style="text-align: left; padding-left: 20px;">
                                    ${precautionsHtml}
                                </ul>
                                <p style="margin-top: 12px; font-size: 0.85em; opacity: 0.8;">
                                    ⚕️ This tool is for preliminary screening only. Always consult a healthcare professional.
                                </p>
                            </div>
                        `;
                        resultContent.innerHTML = html;
                        resultSection.classList.add('active', 'animate__animated', 'animate__fadeInUp');
                    }
                }, 500);
            })
            .catch(error => {
                loadingOverlay.classList.remove('animate__fadeIn');
                loadingOverlay.classList.add('animate__fadeOut');

                setTimeout(() => {
                    loadingOverlay.style.display = 'none';
                    loadingOverlay.classList.remove('animate__fadeOut');
                    alert(`Something went wrong: ${error.message}`);
                }, 500);
            });
    });

    closeResults.addEventListener('click', () => {
        resultSection.classList.remove('active');
        resultContent.innerHTML = '';
    });

    const buttons = document.querySelectorAll('.analyze-btn, .contact-button');
    buttons.forEach(button => {
        button.addEventListener('mousedown', function (e) {
            const x = e.clientX - e.target.getBoundingClientRect().left;
            const y = e.clientY - e.target.getBoundingClientRect().top;
            const ripple = document.createElement('span');
            ripple.classList.add('ripple');
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            this.appendChild(ripple);
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    const animateElements = document.querySelectorAll('.hero h2, .hero p, .hero-stats, .hero-image');
    animateElements.forEach((element, index) => {
        element.classList.add('animate__animated', 'animate__fadeInUp');
        element.style.animationDelay = `${0.3 + (index * 0.2)}s`;
    });

    animateOnScroll();
});
