document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    const fileInput = document.getElementById('file-input');
    const fileSelectName = document.getElementById('noFile');
    const uploadForm = document.getElementById('upload-form');
    const loadingElement = document.getElementById('loading');
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                fileSelectName.textContent = this.files[0].name;
            } else {
                fileSelectName.textContent = 'No file chosen...';
            }
        });
    }
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            uploadForm.style.display = 'none';
            loadingElement.style.display = 'flex';
        });
    }
    
    // Tab navigation for results page
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    if (tabButtons.length > 0) {
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Remove active class from all buttons and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                button.classList.add('active');
                const tabId = button.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });
    }
    
    // Create an animated heartbeat effect for the logo
    const heartbeat = document.querySelector('.fa-heartbeat');
    if (heartbeat) {
        // The animation is already handled by CSS
    }
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Optional: Add a fade-in effect for elements when they come into view
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.feature-card, .info, .stats-card');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if (elementPosition < screenPosition) {
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }
        });
    };
    
    // Initialize the fade-in effect
    elements = document.querySelectorAll('.feature-card, .info, .stats-card');
    elements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    });
    
    window.addEventListener('scroll', animateOnScroll);
    // Trigger once on page load
    animateOnScroll();
});
