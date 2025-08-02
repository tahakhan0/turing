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
        };

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.videoStream.src = `data:image/jpeg;base64,${data.image}`;
            this.handleLiveFeedData(data.notifications);
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
            this.websocketStatusDot.classList.replace('bg-gray-400', 'bg-green-500');
            this.websocketStatusText.textContent = 'CONNECTED';
        } else {
            this.websocketStatusDot.classList.replace('bg-green-500', 'bg-gray-400');
            this.websocketStatusText.textContent = 'DISCONNECTED';
        }
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
                this.addNotification(d, 'medium');
            });
        }

        this.peopleCount.textContent = (data.violations?.length || 0) + (data.unknown_detections?.length || 0);
        this.knownCount.textContent = data.violations?.length || 0;
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

        notificationElement.innerHTML = `
            <div class="flex justify-between items-center text-xs">
                <p class="font-bold text-gray-800">${person}</p>
                <p class="text-gray-500">${timestamp}</p>
            </div>
            <p class="text-xs text-gray-600">Detected in: ${area}</p>
        `;

        this.notificationsContainer.prepend(notificationElement);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MonitoringDashboard();
});