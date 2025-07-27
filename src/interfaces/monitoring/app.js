// Turing Monitoring Dashboard JavaScript

class MonitoringDashboard {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.websocket = null;
        this.userId = null;
        this.isConnected = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadFromLocalStorage();
    }

    initializeElements() {
        // Configuration elements
        this.serviceUrlInput = document.getElementById('service-url');
        this.userIdInput = document.getElementById('user-id');
        this.connectBtn = document.getElementById('connect-btn');
        this.testNotificationBtn = document.getElementById('test-notification-btn');
        this.refreshStatusBtn = document.getElementById('refresh-status-btn');

        // Status elements
        this.websocketStatus = document.getElementById('websocket-status');
        this.websocketText = document.getElementById('websocket-text');
        this.monitoringStatus = document.getElementById('monitoring-status');
        this.monitoringText = document.getElementById('monitoring-text');
        this.geminiStatus = document.getElementById('gemini-status');
        this.geminiText = document.getElementById('gemini-text');
        this.segmentsStatus = document.getElementById('segments-status');
        this.segmentsText = document.getElementById('segments-text');

        // Frame analysis elements
        this.frameUpload = document.getElementById('frame-upload');
        this.analyzeFrameBtn = document.getElementById('analyze-frame-btn');
        this.analyzeWithNotificationsBtn = document.getElementById('analyze-with-notifications-btn');
        this.analyzeWithGeminiBtn = document.getElementById('analyze-with-gemini-btn');
        this.analysisResults = document.getElementById('analysis-results');
        this.analysisOutput = document.getElementById('analysis-output');

        // Notifications
        this.notificationsContainer = document.getElementById('notifications-container');
        this.toastContainer = document.getElementById('toast-container');
    }

    attachEventListeners() {
        this.connectBtn.addEventListener('click', () => this.toggleConnection());
        this.testNotificationBtn.addEventListener('click', () => this.sendTestNotification());
        this.refreshStatusBtn.addEventListener('click', () => this.refreshSystemStatus());

        this.frameUpload.addEventListener('change', () => this.onFrameSelected());
        this.analyzeFrameBtn.addEventListener('click', () => this.analyzeFrame());
        this.analyzeWithNotificationsBtn.addEventListener('click', () => this.analyzeFrameWithNotifications());
        this.analyzeWithGeminiBtn.addEventListener('click', () => this.analyzeFrameWithGemini());

        this.serviceUrlInput.addEventListener('change', () => this.saveToLocalStorage());
        this.userIdInput.addEventListener('change', () => this.saveToLocalStorage());
    }

    loadFromLocalStorage() {
        const savedServiceUrl = localStorage.getItem('turing-service-url');
        const savedUserId = localStorage.getItem('turing-user-id');

        if (savedServiceUrl) {
            this.serviceUrlInput.value = savedServiceUrl;
            this.apiBaseUrl = savedServiceUrl;
        }
        if (savedUserId) {
            this.userIdInput.value = savedUserId;
        }
    }

    saveToLocalStorage() {
        localStorage.setItem('turing-service-url', this.serviceUrlInput.value);
        localStorage.setItem('turing-user-id', this.userIdInput.value);
        this.apiBaseUrl = this.serviceUrlInput.value;
    }

    async toggleConnection() {
        if (this.isConnected) {
            this.disconnect();
        } else {
            await this.connect();
        }
    }

    async connect() {
        this.userId = this.userIdInput.value.trim();
        if (!this.userId) {
            this.showToast('Please enter a User ID', 'error');
            return;
        }

        this.saveToLocalStorage();
        
        try {
            // Test API connection first
            await this.refreshSystemStatus();

            // Connect WebSocket
            const wsUrl = this.apiBaseUrl.replace('http', 'ws') + `/monitoring/notifications/${this.userId}`;
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                this.isConnected = true;
                this.updateConnectionUI();
                this.showToast('Connected successfully', 'success');
                this.updateWebSocketStatus('connected');
            };

            this.websocket.onmessage = (event) => {
                try {
                    const notification = JSON.parse(event.data);
                    this.handleNotification(notification);
                } catch (e) {
                    console.error('Failed to parse notification:', e);
                }
            };

            this.websocket.onclose = () => {
                this.isConnected = false;
                this.updateConnectionUI();
                this.updateWebSocketStatus('disconnected');
                if (this.websocket !== null) {
                    this.showToast('Connection lost', 'error');
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showToast('Connection error', 'error');
                this.updateWebSocketStatus('error');
            };

        } catch (error) {
            console.error('Connection failed:', error);
            this.showToast('Failed to connect: ' + error.message, 'error');
        }
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
        this.updateConnectionUI();
        this.updateWebSocketStatus('disconnected');
        this.showToast('Disconnected', 'info');
    }

    updateConnectionUI() {
        if (this.isConnected) {
            this.connectBtn.textContent = 'Disconnect';
            this.connectBtn.className = 'px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700';
            this.testNotificationBtn.disabled = false;
            this.analyzeFrameBtn.disabled = !this.frameUpload.files.length;
            this.analyzeWithNotificationsBtn.disabled = !this.frameUpload.files.length;
            this.analyzeWithGeminiBtn.disabled = !this.frameUpload.files.length;
        } else {
            this.connectBtn.textContent = 'Connect';
            this.connectBtn.className = 'px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700';
            this.testNotificationBtn.disabled = true;
            this.analyzeFrameBtn.disabled = true;
            this.analyzeWithNotificationsBtn.disabled = true;
            this.analyzeWithGeminiBtn.disabled = true;
        }
    }

    updateWebSocketStatus(status) {
        const statusColors = {
            connected: 'bg-green-500',
            disconnected: 'bg-gray-400',
            error: 'bg-red-500'
        };

        const statusTexts = {
            connected: 'Connected',
            disconnected: 'Disconnected',
            error: 'Error'
        };

        this.websocketStatus.className = `status-dot ${statusColors[status]} mx-auto mb-2`;
        this.websocketText.textContent = statusTexts[status];
    }

    async refreshSystemStatus() {
        try {
            // Check monitoring service
            const monitoringResponse = await fetch(`${this.apiBaseUrl}/monitoring/health`);
            if (monitoringResponse.ok) {
                const monitoringData = await monitoringResponse.json();
                this.monitoringStatus.className = 'status-dot bg-green-500 mx-auto mb-2';
                this.monitoringText.textContent = 'Online';

                // Update Gemini status
                if (monitoringData.gemini_analysis) {
                    const geminiEnabled = monitoringData.gemini_analysis.enabled;
                    this.geminiStatus.className = `status-dot ${geminiEnabled ? 'bg-green-500' : 'bg-yellow-500'} mx-auto mb-2`;
                    this.geminiText.textContent = geminiEnabled ? 'Available' : 'Not configured';
                }
            } else {
                this.monitoringStatus.className = 'status-dot bg-red-500 mx-auto mb-2';
                this.monitoringText.textContent = 'Error';
            }

            // Check user segments
            if (this.userId) {
                const segmentsResponse = await fetch(`${this.apiBaseUrl}/monitoring/segments/user/${this.userId}`);
                if (segmentsResponse.ok) {
                    const segmentsData = await segmentsResponse.json();
                    const segmentCount = segmentsData.total_segments || 0;
                    this.segmentsStatus.className = `status-dot ${segmentCount > 0 ? 'bg-green-500' : 'bg-yellow-500'} mx-auto mb-2`;
                    this.segmentsText.textContent = `${segmentCount} defined`;
                }
            }

        } catch (error) {
            console.error('Status check failed:', error);
            this.monitoringStatus.className = 'status-dot bg-red-500 mx-auto mb-2';
            this.monitoringText.textContent = 'Offline';
        }
    }

    async sendTestNotification() {
        if (!this.userId) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/monitoring/notify/test/${this.userId}`, {
                method: 'POST'
            });

            if (response.ok) {
                this.showToast('Test notification sent', 'success');
            } else {
                this.showToast('Failed to send test notification', 'error');
            }
        } catch (error) {
            this.showToast('Test notification failed: ' + error.message, 'error');
        }
    }

    onFrameSelected() {
        const hasFile = this.frameUpload.files.length > 0;
        if (this.isConnected) {
            this.analyzeFrameBtn.disabled = !hasFile;
            this.analyzeWithNotificationsBtn.disabled = !hasFile;
            this.analyzeWithGeminiBtn.disabled = !hasFile;
        }
    }

    async analyzeFrame() {
        await this.performFrameAnalysis('/monitoring/analyze/frame');
    }

    async analyzeFrameWithNotifications() {
        await this.performFrameAnalysis('/monitoring/analyze/frame/with-notifications');
    }

    async analyzeFrameWithGemini() {
        await this.performFrameAnalysis('/monitoring/analyze/frame/gemini');
    }

    async performFrameAnalysis(endpoint) {
        if (!this.frameUpload.files.length || !this.userId) return;

        const formData = new FormData();
        formData.append('file', this.frameUpload.files[0]);

        try {
            this.showAnalysisLoading();

            const response = await fetch(`${this.apiBaseUrl}${endpoint}?user_id=${this.userId}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayAnalysisResults(result);
            this.showToast('Analysis completed', 'success');

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showToast('Analysis failed: ' + error.message, 'error');
            this.hideAnalysisResults();
        }
    }

    showAnalysisLoading() {
        this.analysisResults.classList.remove('hidden');
        this.analysisOutput.textContent = 'Analyzing frame...';
    }

    displayAnalysisResults(result) {
        this.analysisResults.classList.remove('hidden');
        this.analysisOutput.textContent = JSON.stringify(result, null, 2);
    }

    hideAnalysisResults() {
        this.analysisResults.classList.add('hidden');
    }

    handleNotification(notification) {
        console.log('Received notification:', notification);
        
        // Add to notifications container
        this.addNotificationToContainer(notification);
        
        // Show toast
        const message = notification.title || 'New notification';
        const type = notification.severity === 'high' ? 'error' : 
                    notification.severity === 'medium' ? 'warning' : 'info';
        this.showToast(message, type);
    }

    addNotificationToContainer(notification) {
        // Clear "no notifications" message
        if (this.notificationsContainer.children.length === 1 && 
            this.notificationsContainer.children[0].textContent.includes('No notifications')) {
            this.notificationsContainer.innerHTML = '';
        }

        const notificationElement = document.createElement('div');
        notificationElement.className = `border rounded-lg p-4 ${this.getNotificationBorderColor(notification.severity)}`;
        
        const timestamp = new Date(notification.timestamp).toLocaleTimeString();
        
        notificationElement.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h3 class="font-medium text-gray-900">${notification.title}</h3>
                <span class="text-xs text-gray-500">${timestamp}</span>
            </div>
            <p class="text-sm text-gray-700 mb-3">${notification.message}</p>
            ${notification.data?.analysis_summary ? 
                `<div class="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                    <strong>AI Analysis:</strong> ${notification.data.analysis_summary}
                </div>` : ''
            }
            ${notification.data?.frame_image_url ? 
                `<div class="mt-2">
                    <img src="${notification.data.frame_image_url}" alt="Detection frame" class="max-w-xs rounded border">
                </div>` : ''
            }
        `;

        this.notificationsContainer.insertBefore(notificationElement, this.notificationsContainer.firstChild);

        // Limit to 20 notifications
        while (this.notificationsContainer.children.length > 20) {
            this.notificationsContainer.removeChild(this.notificationsContainer.lastChild);
        }
    }

    getNotificationBorderColor(severity) {
        switch (severity) {
            case 'high': return 'border-red-200 bg-red-50';
            case 'medium': return 'border-yellow-200 bg-yellow-50';
            default: return 'border-blue-200 bg-blue-50';
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `notification max-w-sm p-4 rounded-lg shadow-lg text-white ${this.getToastColor(type)}`;
        toast.textContent = message;

        this.toastContainer.appendChild(toast);

        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 100);

        // Remove after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentNode) {
                    this.toastContainer.removeChild(toast);
                }
            }, 300);
        }, 5000);
    }

    getToastColor(type) {
        switch (type) {
            case 'success': return 'bg-green-600';
            case 'error': return 'bg-red-600';
            case 'warning': return 'bg-yellow-600';
            default: return 'bg-blue-600';
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MonitoringDashboard();
});