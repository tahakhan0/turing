// Alan AI Setup Completion Page - Main Application JavaScript

class CompletionPage {
    constructor() {
        this.currentVideoPath = null;
        this.currentUserId = null;
        this.apiBaseUrl = 'http://localhost:8000';
        this.countdownTime = 5;
        this.countdownInterval = null;
        
        this.initializeElements();
        this.loadDataFromUrl();
        this.displaySummaryStats();
        this.startCountdown();
        this.createAdditionalConfetti();
    }

    initializeElements() {
        this.countdownElement = document.getElementById('countdown');
        this.skipButton = document.getElementById('skip-redirect');
        this.peopleCountElement = document.getElementById('people-count');
        this.areasCountElement = document.getElementById('areas-count');
        this.permissionsCountElement = document.getElementById('permissions-count');

        // Event listeners
        this.skipButton.addEventListener('click', () => this.navigateToMonitoring());
    }

    loadDataFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        this.currentVideoPath = urlParams.get('video_path') || '';
        this.currentUserId = urlParams.get('user_id') || '';
        const serviceUrl = urlParams.get('service_url');
        
        if (serviceUrl) {
            this.apiBaseUrl = serviceUrl;
        }

        // Get summary stats from URL params if available
        this.peopleCount = parseInt(urlParams.get('people_count')) || 0;
        this.areasCount = parseInt(urlParams.get('areas_count')) || 0;
        this.permissionsCount = parseInt(urlParams.get('permissions_count')) || 0;
    }

    displaySummaryStats() {
        // Animate the numbers counting up
        this.animateCounter(this.peopleCountElement, this.peopleCount, 1000);
        this.animateCounter(this.areasCountElement, this.areasCount, 1200);
        this.animateCounter(this.permissionsCountElement, this.permissionsCount, 1400);
    }

    animateCounter(element, targetValue, duration) {
        let currentValue = 0;
        const increment = targetValue / (duration / 50);
        
        const timer = setInterval(() => {
            currentValue += increment;
            if (currentValue >= targetValue) {
                element.textContent = targetValue;
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(currentValue);
            }
        }, 50);
    }

    startCountdown() {
        this.countdownElement.textContent = this.countdownTime;
        
        this.countdownInterval = setInterval(() => {
            this.countdownTime--;
            this.countdownElement.textContent = this.countdownTime;
            
            if (this.countdownTime <= 0) {
                clearInterval(this.countdownInterval);
                this.navigateToMonitoring();
            }
        }, 1000);
    }

    navigateToMonitoring() {
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
        }

        // Add fade out effect
        document.body.style.transition = 'opacity 0.5s ease-out';
        document.body.style.opacity = '0';
        
        setTimeout(() => {
            const params = new URLSearchParams({
                video_path: this.currentVideoPath,
                user_id: this.currentUserId,
                service_url: this.apiBaseUrl
            });
            window.location.href = `../monitoring/index.html?${params.toString()}`;
        }, 500);
    }

    createAdditionalConfetti() {
        // Create more dynamic confetti pieces
        for (let i = 0; i < 20; i++) {
            setTimeout(() => {
                this.createConfettiPiece();
            }, i * 100);
        }

        // Continue creating confetti every 2 seconds
        setInterval(() => {
            for (let i = 0; i < 5; i++) {
                setTimeout(() => {
                    this.createConfettiPiece();
                }, i * 200);
            }
        }, 2000);
    }

    createConfettiPiece() {
        const confetti = document.createElement('div');
        const colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7',
            '#fd79a8', '#6c5ce7', '#a29bfe', '#00b894', '#e17055',
            '#0984e3', '#00cec9', '#fdcb6e', '#e84393'
        ];
        
        confetti.className = 'confetti';
        confetti.style.left = Math.random() * 100 + '%';
        confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
        confetti.style.animationDuration = (Math.random() * 2 + 2) + 's';
        confetti.style.animationDelay = Math.random() * 1 + 's';
        
        // Random shapes
        if (Math.random() > 0.5) {
            confetti.style.borderRadius = '50%';
        } else {
            confetti.style.transform = 'rotate(45deg)';
        }
        
        document.body.appendChild(confetti);
        
        // Remove confetti after animation
        setTimeout(() => {
            if (document.body.contains(confetti)) {
                document.body.removeChild(confetti);
            }
        }, 4000);
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CompletionPage();
});