// Turing Monitoring Dashboard JavaScript

class MonitoringDashboard {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.websocket = null;
        this.userId = null;
        this.isLiveMonitoring = false;

        this.initializeElements();
        this.loadFromUrlParams();
    }

    initializeElements() {
        // Status
        this.websocketStatusDot = document.getElementById('websocket-status-dot');
        this.websocketStatusText = document.getElementById('websocket-status-text');

        // Detections
        this.peopleCount = document.getElementById('people-count');
        this.knownCount = document.getElementById('known-count');
        this.violationsCount = document.getElementById('violations-count');

        // Event Log
        this.notificationsContainer = document.getElementById('notifications-container');
        
        // Video Stream
        this.videoStream = document.getElementById('video-stream');
    }

    loadFromUrlParams() {
        const urlParams = new URLSearchParams(window.location.search);
        this.userId = urlParams.get('user_id');
        const serviceUrl = urlParams.get('service_url');
        
        if (serviceUrl) {
            this.apiBaseUrl = serviceUrl;
        }

        if (this.userId) {
            this.startLiveMonitoring();
        } else {
            alert('User ID not found in URL. Please provide a user_id parameter.');
        }
    }

    startLiveMonitoring() {
        if (this.isLiveMonitoring) return;

        const wsUrl = `${this.apiBaseUrl.replace('http', 'ws')}/monitoring/ws/live_feed/${this.userId}`;
        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            this.isLiveMonitoring = true;
            this.updateConnectionStatus(true);
            this.notificationsContainer.innerHTML = ''; // Clear previous logs
            this.initializeCanvas(); // Initialize canvas when connection opens
        };

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.image) {
                this.updateVideoCanvas(data.image);
            }
            if (data.notifications) {
                console.log('Analysis result:', data.notifications);
                this.handleLiveFeedData(data.notifications);
            }
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };

        this.websocket.onclose = () => {
            this.isLiveMonitoring = false;
            this.updateConnectionStatus(false);
        };
    }

    updateConnectionStatus(connected) {
        this.isLiveMonitoring = connected;
        if (connected) {
            this.websocketStatusDot.classList.remove('bg-red-400');
            this.websocketStatusDot.classList.add('bg-green-500');
            this.websocketStatusText.textContent = 'Connected';
        } else {
            this.websocketStatusDot.classList.remove('bg-green-500');
            this.websocketStatusDot.classList.add('bg-red-400');
            this.websocketStatusText.textContent = 'Disconnected';
        }
    }

    initializeCanvas() {
        const canvas = this.videoStream;
        const container = canvas.parentElement;
        const containerRect = container.getBoundingClientRect();
        
        if (containerRect.width === 0 || containerRect.height === 0) {
            canvas.width = 640;
            canvas.height = 360;
        } else {
            canvas.width = containerRect.width;
            canvas.height = containerRect.height;
        }
        
        // Test canvas visibility by drawing a red rectangle
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'red';
        ctx.fillRect(0, 0, 100, 100);
        
        console.log('Canvas initialized with dimensions:', canvas.width, 'x', canvas.height);
        console.log('Canvas element:', canvas);
        console.log('Canvas style:', window.getComputedStyle(canvas));
    }

    updateVideoCanvas(base64Image) {
        const canvas = this.videoStream;
        if (!canvas) {
            console.error('Canvas element not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error('Failed to get 2D context');
            return;
        }
        
        // Create an image from the base64 data
        const img = new Image();
        img.onload = () => {
            // Clear the entire canvas first
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw the image to fill the entire canvas
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Force a repaint
            canvas.style.display = 'none';
            canvas.offsetHeight; // Trigger reflow
            canvas.style.display = '';
            
            console.log('Image updated on canvas');
        };
        img.onerror = (error) => {
            console.error('Image loading failed:', error);
        };
        img.src = `data:image/jpeg;base64,${base64Image}`;
    }

    handleLiveFeedData(data) {
        if (!data) return; // Add this check to prevent the error

        let violationCount = 0;
        if (data.violations) {
            data.violations.forEach(v => {
                this.addNotification(v, 'high');
                violationCount++;
            });
        }
        if (data.unknown_detections) {
            data.unknown_detections.forEach(d => {
                // Unknown persons are high-severity since non-residents aren't allowed
                const severity = d.type === 'unauthorized_unknown_person' ? 'high' : 'medium';
                this.addNotification(d, severity);
            });
        }

        // Total people detected
        const totalPeople = (data.violations?.length || 0) + (data.unknown_detections?.length || 0);
        this.peopleCount.textContent = totalPeople;
        
        // Known people (violations are from known people)
        this.knownCount.textContent = data.violations?.length || 0;
        
        // Actual violations (not just known people)
        this.violationsCount.textContent = violationCount;
    }

    addNotification(item, severity) {
        const notificationElement = document.createElement('div');
        notificationElement.className = `notification-item p-3 rounded-md border-l-4`
        if (severity === 'high') {
            notificationElement.classList.add('border-red-500', 'bg-red-50');
        } else if (severity === 'medium') {
            notificationElement.classList.add('border-yellow-500', 'bg-yellow-50');
        } else {
            notificationElement.classList.add('border-blue-500', 'bg-blue-50');
        }

        const timestamp = new Date(item.timestamp).toLocaleTimeString();
        const person = item.person_name || 'Unknown';
        const area = item.segment?.area_type || 'N/A';
        const geminiAnalysis = item.gemini_analysis;
        const violationReason = item.violation_reason;

        // Create notification content
        let content = `
            <div class="flex justify-between items-center text-xs">
                <p class="font-bold text-gray-800">${person}</p>
                <p class="text-gray-500">${timestamp}</p>
            </div>
            <p class="text-xs text-gray-600">Detected in: ${area}</p>
        `;

        // Add violation reason if present
        if (violationReason) {
            content += `<p class="text-xs text-red-600 font-medium mt-1">⚠️ ${violationReason}</p>`;
        }

        // Add Gemini AI analysis if available
        if (geminiAnalysis) {
            content += `
                <div class="mt-2 p-2 bg-gray-100 rounded text-xs">
                    <p class="text-gray-500 font-medium mb-1">🤖 AI Analysis:</p>
                    <p class="text-gray-700 italic">${geminiAnalysis}</p>
                </div>
            `;
        }

        notificationElement.innerHTML = content;
        this.notificationsContainer.prepend(notificationElement);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MonitoringDashboard();
});