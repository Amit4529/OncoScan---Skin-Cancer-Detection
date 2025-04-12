document.addEventListener('DOMContentLoaded', function() {
    const uploadContainer = document.getElementById('upload-container');
    const imageUpload = document.getElementById('image-upload');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const imagePreview = document.getElementById('image-preview');
    const scanForm = document.getElementById('scanForm');
    const loadingOverlay = document.getElementById('loading-overlay');

    // Counter animation for stats
    const counters = document.querySelectorAll('.counter');
    const speed = 200;

    // Intersection Observer for animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                if (entry.target.classList.contains('counter')) {
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
                }
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(counter => {
        observer.observe(counter);
    });

    // Animate elements when they come into view
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

    // Handle file upload and preview with animation
    imageUpload.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();

            reader.onload = function(e) {
                uploadPlaceholder.style.display = 'none';
                imagePreview.style.display = 'block';

                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('animate__animated', 'animate__fadeIn');

                imagePreview.innerHTML = '';
                imagePreview.appendChild(img);
            }

            reader.readAsDataURL(file);
        }
    });

    // Drag and drop functionality
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

    // Submit form with real API call
    scanForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;
        const image = document.getElementById('image-upload').files[0];

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
        formData.append('image', image);

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
                    alert(`Result: ${data.result}\nConfidence Score: ${(data.score * 100).toFixed(2)}%`);
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

    // Ripple effect
    const buttons = document.querySelectorAll('.analyze-btn, .contact-button');
    buttons.forEach(button => {
        button.addEventListener('mousedown', function(e) {
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

    // Animate hero elements
    const animateElements = document.querySelectorAll('.hero h2, .hero p, .hero-stats, .hero-image');
    animateElements.forEach((element, index) => {
        element.classList.add('animate__animated', 'animate__fadeInUp');
        element.style.animationDelay = `${0.3 + (index * 0.2)}s`;
    });

    animateOnScroll();
});
