// Turing Area Segmentation Web UI - Main Application JavaScript

class AreaSegmentationUI {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentVideoPath = null;
        this.currentUserId = null;
        this.labeledFaces = [];
        this.segmentationData = null;
        this.areasData = [];
        this.verifiedAreas = new Set();
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadDataFromUrl();
        this.checkServiceConnection();
    }

    initializeElements() {
        // Form elements
        this.frameNumberInput = document.getElementById('frame-number');
        this.randomFrameBtn = document.getElementById('random-frame');
        this.boxThresholdSlider = document.getElementById('box-threshold');
        this.textThresholdSlider = document.getElementById('text-threshold');
        this.boxThresholdValue = document.getElementById('box-threshold-value');
        this.textThresholdValue = document.getElementById('text-threshold-value');
        this.serviceUrlInput = document.getElementById('service-url');
        this.startSegmentationBtn = document.getElementById('start-segmentation');
        this.segmentText = document.getElementById('segment-text');
        this.segmentSpinner = document.getElementById('segment-spinner');

        // Navigation elements
        this.backToFacesBtn = document.getElementById('back-to-faces');
        this.continueBtn = document.getElementById('continue-to-permissions');
        this.proceedBtn = document.getElementById('proceed-to-permissions');

        // Info display elements
        this.currentVideoPathElement = document.getElementById('current-video-path');
        this.currentUserIdElement = document.getElementById('current-user-id');
        this.labeledFacesCountElement = document.getElementById('labeled-faces-count');

        // Results elements
        this.setupSection = document.getElementById('setup-section');
        this.resultsSection = document.getElementById('results-section');
        this.loadingState = document.getElementById('loading-state');
        this.errorState = document.getElementById('error-state');
        this.errorMessage = document.getElementById('error-message');

        // Summary elements
        this.totalAreas = document.getElementById('total-areas');
        this.verifiedAreasCount = document.getElementById('verified-areas');
        this.areaTypes = document.getElementById('area-types');
        this.totalAreaSize = document.getElementById('total-area-size');

        // Frame visualization elements (removed in unified design)
        // this.frameImage = document.getElementById('frame-image');
        // this.segmentationOverlay = document.getElementById('segmentation-overlay');
        // this.showOriginalBtn = document.getElementById('show-original');
        // this.showSegmentedBtn = document.getElementById('show-segmented');

        // Segmented images elements
        this.segmentedImagesContainer = document.getElementById('segmented-images-container');
        this.filterAreaType = document.getElementById('filter-area-type');
        this.filterVerification = document.getElementById('filter-verification');
        this.viewMode = document.getElementById('view-mode');
        this.verifyAllBtn = document.getElementById('verify-all');

        // Remove modal elements as we're using inline verification now
        
        // Auto-processing notice
        this.autoProcessingNotice = document.getElementById('auto-processing-notice');
    }

    attachEventListeners() {
        // Navigation
        this.backToFacesBtn.addEventListener('click', () => this.navigateToFaceRecognition());
        this.continueBtn.addEventListener('click', () => this.navigateToPermissions());
        this.proceedBtn.addEventListener('click', () => this.navigateToPermissions());

        // Threshold sliders
        this.boxThresholdSlider.addEventListener('input', (e) => {
            this.boxThresholdValue.textContent = e.target.value;
        });
        this.textThresholdSlider.addEventListener('input', (e) => {
            this.textThresholdValue.textContent = e.target.value;
        });

        // Frame selection
        this.randomFrameBtn.addEventListener('click', () => this.selectRandomFrame());

        // Segmentation
        this.startSegmentationBtn.addEventListener('click', () => this.startSegmentation());

        // Visualization controls (removed in unified design)
        // this.showOriginalBtn.addEventListener('click', () => this.showOriginalFrame());
        // this.showSegmentedBtn.addEventListener('click', () => this.showSegmentedFrame());

        // Verification
        this.verifyAllBtn.addEventListener('click', () => this.verifyAllAreas());

        // Filters and view mode
        this.filterAreaType.addEventListener('change', () => this.applyFilters());
        this.filterVerification.addEventListener('change', () => this.applyFilters());
        this.viewMode.addEventListener('change', () => this.displaySegmentedImages());

        // Remove modal event listeners as we're using inline verification
    }

    loadDataFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        this.currentVideoPath = urlParams.get('video_path');
        this.currentUserId = urlParams.get('user_id');
        const facesCount = urlParams.get('faces_count') || '0';
        const autoStart = urlParams.get('auto_start') === 'true';

        if (this.currentVideoPath) {
            this.currentVideoPathElement.textContent = this.currentVideoPath;
        }
        if (this.currentUserId) {
            this.currentUserIdElement.textContent = this.currentUserId;
        }
        this.labeledFacesCountElement.textContent = facesCount;

        // Update API base URL if provided
        const serviceUrl = urlParams.get('service_url');
        if (serviceUrl) {
            this.apiBaseUrl = serviceUrl;
            this.serviceUrlInput.value = serviceUrl;
        }

        // Debug logging to help troubleshoot
        console.log('URL Parameters:', {
            autoStart,
            currentUserId: this.currentUserId,
            facesCount,
            facesCountInt: parseInt(facesCount)
        });

        // Auto-start segmentation if coming from face recognition with auto_start flag
        if (autoStart && this.currentUserId && facesCount && parseInt(facesCount) > 0) {
            console.log('Auto-starting segmentation...');
            // Hide setup section immediately and show loading
            this.setupSection.classList.add('hidden');
            this.setLoadingState(true);
            
            // Start segmentation immediately
            setTimeout(() => {
                this.autoStartSegmentation();
            }, 500);
        } else {
            console.log('Not auto-starting. Showing setup section.');
        }
    }

    async checkServiceConnection() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/segmentation/health`);
            if (response.ok) {
                this.updateConnectionStatus(true);
            } else {
                this.updateConnectionStatus(false);
            }
        } catch (error) {
            this.updateConnectionStatus(false);
        }
    }

    updateConnectionStatus(connected) {
        const statusIndicator = document.querySelector('#connection-status .w-2');
        const statusText = document.querySelector('#connection-status span');
        
        if (connected) {
            statusIndicator.className = 'w-2 h-2 bg-green-400 rounded-full';
            statusText.textContent = 'Connected';
        } else {
            statusIndicator.className = 'w-2 h-2 bg-red-400 rounded-full';
            statusText.textContent = 'Disconnected';
        }
    }

    selectRandomFrame() {
        // Generate random frame number between 1 and 1000
        const randomFrame = Math.floor(Math.random() * 1000) + 1;
        this.frameNumberInput.value = randomFrame;
    }

    async autoStartSegmentation() {
        // Auto-start segmentation with a user-friendly message
        console.log('Auto-starting segmentation after face recognition completion...');
        
        // Show auto-processing notice
        if (this.autoProcessingNotice) {
            this.autoProcessingNotice.classList.remove('hidden');
        }
        
        // Update the UI to show it's automatically starting
        this.segmentText.textContent = 'Auto-starting segmentation...';
        this.startSegmentationBtn.classList.add('bg-green-600');
        this.startSegmentationBtn.classList.remove('bg-brand-purple');
        
        // Start the segmentation process directly
        await this.startSegmentation();
        
        // Hide the notice after processing
        if (this.autoProcessingNotice) {
            this.autoProcessingNotice.classList.add('hidden');
        }
    }

    async startSegmentation() {
        if (!this.currentUserId) {
            this.showError('Missing user ID. Please go back to face recognition first.');
            return;
        }

        this.setLoadingState(true);
        this.hideError();

        // Start the segmentation process in the background
        fetch(`${this.apiBaseUrl}/segmentation/segment/frames/${this.currentUserId}`, {
            method: 'POST'
        }).then(response => {
            if (!response.ok) {
                // Handle errors that occur after the polling has started
                response.json().then(errorData => {
                    this.showError(errorData.detail || 'Segmentation failed');
                    this.setLoadingState(false);
                    clearInterval(this.pollingInterval);
                });
            }
        });

        // Start polling for results after a delay
        setTimeout(() => {
            this.pollingInterval = setInterval(() => this.pollForSegmentationResults(), 2000);
        }, 5000);
    }

    async pollForSegmentationResults() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/segmentation/visualizations/${this.currentUserId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.visualizations && data.visualizations.length > 0) {
                    this.displayPartialResults(data.visualizations);
                }
            }

            // Check if the full results are ready
            const fullResultsResponse = await fetch(`${this.apiBaseUrl}/segmentation/user/${this.currentUserId}`);
            if (fullResultsResponse.ok) {
                const fullData = await fullResultsResponse.json();
                if (fullData.segmentation_data && fullData.segmentation_data.segments && fullData.segmentation_data.segments.length > 0) {
                    clearInterval(this.pollingInterval);
                    this.segmentationData = fullData.segmentation_data;
                    this.areasData = this.segmentationData.segments || [];
                    this.displayResults(null, true);
                    this.setLoadingState(false);
                }
            }
        } catch (error) {
            console.error('Polling error:', error);
            this.showError('Failed to fetch segmentation results.');
            this.setLoadingState(false);
            clearInterval(this.pollingInterval);
        }
    }

    displayPartialResults(visualizationUrls) {
        this.setupSection.classList.add('hidden');
        this.resultsSection.classList.remove('hidden');
        this.segmentedImagesContainer.innerHTML = '';

        const gridContainer = document.createElement('div');
        gridContainer.className = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6';

        visualizationUrls.forEach(url => {
            const imageCard = this.createPartialImageCard(url);
            gridContainer.appendChild(imageCard);
        });

        this.segmentedImagesContainer.appendChild(gridContainer);
    }

    createPartialImageCard(imageUrl) {
        const card = document.createElement('div');
        card.className = 'bg-white border-2 border-gray-300 rounded-lg overflow-hidden';
        card.innerHTML = `
            <div class="relative">
                <img src="${this.apiBaseUrl}${imageUrl}" alt="Segmentation in progress" class="w-full h-48 object-cover">
                <div class="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                    <p class="text-white text-lg font-semibold">Processing...</p>
                </div>
            </div>
            <div class="p-4">
                <h3 class="font-semibold text-gray-900 capitalize">Segmenting...</h3>
                <div class="text-xs text-gray-600 mb-3">
                    <p>Please wait while we analyze this image.</p>
                </div>
            </div>
        `;
        return card;
    }

    displayResults(imagePath, isFromExtractedFrames = false) {
        this.setupSection.classList.add('hidden');
        this.resultsSection.classList.remove('hidden');

        // Update summary
        this.updateSummary();

        // Display segmented images instead of single frame
        this.displaySegmentedImages();

        // Enable continue/proceed buttons if areas are detected
        if (this.areasData.length > 0) {
            this.continueBtn.disabled = false;
            this.proceedBtn.disabled = false;
        }
    }

    updateSummary() {
        const areaTypesSet = new Set(this.areasData.map(area => area.area_type || area.label || 'unknown'));
        const totalArea = this.areasData.reduce((sum, area) => sum + (area.dimensions?.area || 0), 0);

        this.totalAreas.textContent = this.areasData.length;
        this.verifiedAreasCount.textContent = this.verifiedAreas.size;
        this.areaTypes.textContent = areaTypesSet.size;
        this.totalAreaSize.textContent = totalArea.toFixed(1);
    }

    drawSegmentationOverlay() {
        const canvas = this.segmentationOverlay;
        const ctx = canvas.getContext('2d');
        const img = this.frameImage;

        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        canvas.style.width = img.clientWidth + 'px';
        canvas.style.height = img.clientHeight + 'px';

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const scaleX = canvas.width / img.naturalWidth;
        const scaleY = canvas.height / img.naturalHeight;

        const colors = [
            '#ef4444', '#f97316', '#eab308', '#22c55e', 
            '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899'
        ];

        this.areasData.forEach((area, index) => {
            // Handle both polygon and bbox formats
            if (area.bbox) {
                // Draw bounding box for Replicate API results
                const color = colors[index % colors.length];
                ctx.strokeStyle = color;
                ctx.fillStyle = color + '40'; // Add transparency
                ctx.lineWidth = 2;

                const x1 = area.bbox.x1 * scaleX;
                const y1 = area.bbox.y1 * scaleY;
                const x2 = area.bbox.x2 * scaleX;
                const y2 = area.bbox.y2 * scaleY;

                ctx.beginPath();
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.fill();
                ctx.stroke();

                // Add label
                const centerX = (x1 + x2) / 2;
                const centerY = (y1 + y2) / 2;
                
                ctx.fillStyle = color;
                ctx.font = '14px Inter, sans-serif';
                ctx.fontWeight = 'bold';
                ctx.textAlign = 'center';
                ctx.fillText((area.area_type || area.label || 'unknown').replace('_', ' ').toUpperCase(), centerX, centerY);
            } else if (area.polygon && area.polygon.length > 0) {
                // Draw polygon for other API results
                const color = colors[index % colors.length];
                ctx.strokeStyle = color;
                ctx.fillStyle = color + '40'; // Add transparency
                ctx.lineWidth = 2;

                ctx.beginPath();
                area.polygon.forEach((point, i) => {
                    const x = point[0] * scaleX;
                    const y = point[1] * scaleY;
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                });
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // Add label
                const centerX = area.polygon.reduce((sum, p) => sum + p[0], 0) / area.polygon.length * scaleX;
                const centerY = area.polygon.reduce((sum, p) => sum + p[1], 0) / area.polygon.length * scaleY;
                
                ctx.fillStyle = color;
                ctx.font = '14px Inter, sans-serif';
                ctx.fontWeight = 'bold';
                ctx.textAlign = 'center';
                ctx.fillText((area.area_type || area.label || 'unknown').replace('_', ' ').toUpperCase(), centerX, centerY);
            }
        });
    }

    displaySegmentedImages() {
        this.segmentedImagesContainer.innerHTML = '';

        const filteredAreas = this.getFilteredAreas();
        const viewMode = this.viewMode.value;

        if (viewMode === 'grid') {
            this.displayGridView(filteredAreas);
        } else {
            this.displayListView(filteredAreas);
        }
    }

    displayGridView(areas) {
        const gridContainer = document.createElement('div');
        gridContainer.className = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6';

        areas.forEach((area, index) => {
            const imageCard = this.createSegmentedImageCard(area, index);
            gridContainer.appendChild(imageCard);
        });

        this.segmentedImagesContainer.appendChild(gridContainer);
    }

    displayListView(areas) {
        const listContainer = document.createElement('div');
        listContainer.className = 'space-y-4';

        areas.forEach((area, index) => {
            const imageCard = this.createSegmentedImageCard(area, index, true);
            listContainer.appendChild(imageCard);
        });

        this.segmentedImagesContainer.appendChild(listContainer);
    }

    createSegmentedImageCard(area, index, isListView = false) {
        const colors = [
            'border-red-300', 'border-orange-300', 'border-yellow-300', 'border-green-300',
            'border-cyan-300', 'border-blue-300', 'border-purple-300', 'border-pink-300'
        ];

        const isVerified = this.verifiedAreas.has(area.area_id);
        const borderColor = colors[index % colors.length];
        
        const card = document.createElement('div');
        const cardClasses = isListView 
            ? `segmented-image-card flex bg-white border-2 ${borderColor} rounded-lg overflow-hidden ${isVerified ? 'ring-2 ring-green-500' : ''}` 
            : `segmented-image-card bg-white border-2 ${borderColor} rounded-lg overflow-hidden ${isVerified ? 'ring-2 ring-green-500' : ''}`;
        
        card.className = cardClasses;
        
        const areaType = area.area_type || area.label || 'unknown';
        const confidence = area.confidence || 0;
        
        // Get image source - prefer visualization if available
        let imageSrc = this.getImageSourceForArea(area);
        
        if (isListView) {
            card.innerHTML = `
                <div class="w-48 h-32 flex-shrink-0 relative">
                    <img src="${imageSrc}" alt="${areaType} segment" class="w-full h-full object-cover">
                    <canvas class="absolute inset-0 w-full h-full pointer-events-none" data-area-id="${area.area_id}"></canvas>
                    ${isVerified ? '<div class="verification-badge bg-green-500 text-white">✓ Verified</div>' : '<div class="verification-badge bg-gray-500 text-white">Unverified</div>'}
                </div>
                <div class="flex-1 p-4">
                    <div class="flex justify-between items-start mb-2">
                        <h3 class="text-lg font-semibold text-gray-900 capitalize">${areaType.replace('_', ' ')}</h3>
                        <span class="text-sm text-gray-500">${(confidence * 100).toFixed(1)}% confidence</span>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-4 text-sm text-gray-600 mb-4">
                        <div><span class="font-medium">Type:</span> ${areaType}</div>
                        <div><span class="font-medium">Source:</span> ${area.source_frame || 'N/A'}</div>
                        <div><span class="font-medium">Detection:</span> ${area.source || 'Replicate'}</div>
                        <div><span class="font-medium">Area ID:</span> ${area.area_id}</div>
                    </div>
                    
                    <div class="inline-verification">
                        ${!isVerified ? `
                            <button class="verify-btn bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md font-medium transition-colors">
                                ✓ Verify
                            </button>
                            <button class="reject-btn bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md font-medium transition-colors">
                                ✗ Reject
                            </button>
                        ` : `
                            <div class="flex items-center text-green-600">
                                <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                </svg>
                                <span class="font-medium">Verified</span>
                            </div>
                        `}
                    </div>
                </div>
            `;
        } else {
            card.innerHTML = `
                <div class="relative">
                    <img src="${imageSrc}" alt="${areaType} segment" class="w-full h-48 object-cover">
                    <canvas class="absolute inset-0 w-full h-full pointer-events-none" data-area-id="${area.area_id}"></canvas>
                    ${isVerified ? '<div class="verification-badge bg-green-500 text-white">✓ Verified</div>' : '<div class="verification-badge bg-gray-500 text-white">Unverified</div>'}
                </div>
                <div class="p-4">
                    <div class="flex justify-between items-start mb-2">
                        <h3 class="font-semibold text-gray-900 capitalize">${areaType.replace('_', ' ')}</h3>
                        <span class="text-xs text-gray-500">${(confidence * 100).toFixed(1)}%</span>
                    </div>
                    
                    <div class="text-xs text-gray-600 mb-3">
                        <div>Source: ${area.source_frame || 'N/A'}</div>
                        <div>ID: ${area.area_id}</div>
                    </div>
                    
                    <div class="inline-verification">
                        ${!isVerified ? `
                            <button class="verify-btn bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm font-medium transition-colors">
                                ✓ Verify
                            </button>
                            <button class="reject-btn bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm font-medium transition-colors">
                                ✗ Reject
                            </button>
                        ` : `
                            <div class="flex items-center text-green-600 text-sm">
                                <svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                </svg>
                                <span>Verified</span>
                            </div>
                        `}
                    </div>
                </div>
            `;
        }
        
        // Add event listeners for verification buttons
        if (!isVerified) {
            const verifyBtn = card.querySelector('.verify-btn');
            const rejectBtn = card.querySelector('.reject-btn');
            
            if (verifyBtn) {
                verifyBtn.addEventListener('click', () => this.verifyArea(area.area_id, true));
            }
            
            if (rejectBtn) {
                rejectBtn.addEventListener('click', () => this.verifyArea(area.area_id, false));
            }
        }
        
        // Draw segmentation overlay on the canvas after the image loads
        const img = card.querySelector('img');
        const canvas = card.querySelector('canvas');
        
        if (img && canvas) {
            img.onload = () => {
                this.drawSegmentOverlayOnCard(canvas, area, img);
            };
        }
        
        return card;
    }

    getImageSourceForArea(area) {
        // Try to get the best image source for this area
        
        // If we have frame details with visualization URLs, use that
        if (this.segmentationData.frame_details) {
            const frameDetail = this.segmentationData.frame_details.find(f => 
                f.frame_file === area.source_frame && f.visualization_url
            );
            if (frameDetail && frameDetail.visualization_url) {
                return frameDetail.visualization_url;
            }
        }
        
        // If we have a visualization URL in segmentation data, use that
        if (this.segmentationData.visualization_url) {
            return this.segmentationData.visualization_url;
        }
        
        // If we have source frame path, construct the URL with user_id
        if (area.source_frame_path) {
            return `${this.apiBaseUrl}/static/face_recognition/${this.currentUserId}/extracted_frames/${area.source_frame_path.split('/').pop()}`;
        }
        
        // If we have source frame name, construct the URL with user_id
        if (area.source_frame) {
            return `${this.apiBaseUrl}/static/face_recognition/${this.currentUserId}/extracted_frames/${area.source_frame}`;
        }
        
        // Fallback to image path from segmentation data
        if (this.segmentationData.image_path) {
            return this.segmentationData.image_path;
        }
        
        // Last resort - empty image
        return 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg==';
    }

    drawSegmentOverlayOnCard(canvas, area, img) {
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to match image
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate scale factors
        const scaleX = canvas.width / img.naturalWidth;
        const scaleY = canvas.height / img.naturalHeight;
        
        // Set drawing style
        ctx.strokeStyle = '#ef4444';
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.lineWidth = 2;
        
        // Draw segmentation area
        if (area.bbox) {
            // Draw bounding box
            const x1 = area.bbox.x1 * scaleX;
            const y1 = area.bbox.y1 * scaleY;
            const x2 = area.bbox.x2 * scaleX;
            const y2 = area.bbox.y2 * scaleY;
            
            ctx.beginPath();
            ctx.rect(x1, y1, x2 - x1, y2 - y1);
            ctx.fill();
            ctx.stroke();
            
            // Add label
            const centerX = (x1 + x2) / 2;
            const centerY = (y1 + y2) / 2;
            
            ctx.fillStyle = '#ef4444';
            ctx.font = 'bold 12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText((area.area_type || area.label || 'unknown').replace('_', ' ').toUpperCase(), centerX, centerY);
        } else if (area.polygon && area.polygon.length > 0) {
            // Draw polygon
            ctx.beginPath();
            area.polygon.forEach((point, i) => {
                const x = point[0] * scaleX;
                const y = point[1] * scaleY;
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            
            // Add label
            const centerX = area.polygon.reduce((sum, p) => sum + p[0], 0) / area.polygon.length * scaleX;
            const centerY = area.polygon.reduce((sum, p) => sum + p[1], 0) / area.polygon.length * scaleY;
            
            ctx.fillStyle = '#ef4444';
            ctx.font = 'bold 12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText((area.area_type || area.label || 'unknown').replace('_', ' ').toUpperCase(), centerX, centerY);
        }
    }

    getFilteredAreas() {
        let filtered = [...this.areasData];

        // Filter by area type
        const areaTypeFilter = this.filterAreaType.value;
        if (areaTypeFilter !== 'all') {
            filtered = filtered.filter(area => {
                const areaType = area.area_type || area.label || 'unknown';
                return areaType === areaTypeFilter;
            });
        }

        // Filter by verification status
        const verificationFilter = this.filterVerification.value;
        if (verificationFilter === 'verified') {
            filtered = filtered.filter(area => this.verifiedAreas.has(area.area_id));
        } else if (verificationFilter === 'unverified') {
            filtered = filtered.filter(area => !this.verifiedAreas.has(area.area_id));
        }

        return filtered;
    }

    applyFilters() {
        this.displaySegmentedImages();
    }

    async verifyArea(areaId, approved) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/segmentation/segment/verify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    area_id: areaId,
                    user_id: this.currentUserId,
                    approved: approved
                })
            });

            if (!response.ok) {
                throw new Error(`Verification failed: ${response.statusText}`);
            }

            const result = await response.json();

            if (approved) {
                this.verifiedAreas.add(areaId);
            } else {
                // Remove from areas data if rejected
                this.areasData = this.areasData.filter(area => area.area_id !== areaId);
            }

            this.updateSummary();
            this.displaySegmentedImages();
            
            // Update proceed button state
            this.updateProceedButtonState();

        } catch (error) {
            console.error('Verification error:', error);
            this.showError(error.message);
        }
    }
    
    updateProceedButtonState() {
        const hasVerifiedAreas = this.verifiedAreas.size > 0;
        this.proceedBtn.disabled = !hasVerifiedAreas;
        this.continueBtn.disabled = !hasVerifiedAreas;
    }




    async verifyAllAreas() {
        const unverifiedAreas = this.areasData.filter(area => !this.verifiedAreas.has(area.area_id));
        
        for (const area of unverifiedAreas) {
            try {
                await fetch(`${this.apiBaseUrl}/segmentation/segment/verify`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        area_id: area.area_id,
                        user_id: this.currentUserId,
                        approved: true
                    })
                });

                this.verifiedAreas.add(area.area_id);
            } catch (error) {
                console.error('Error verifying area:', area.area_id, error);
            }
        }

        this.updateSummary();
        this.displaySegmentedImages();
        
        // Update proceed button state
        this.updateProceedButtonState();
    }

    showOriginalFrame() {
        this.segmentationOverlay.style.display = 'none';
        this.showOriginalBtn.classList.add('bg-brand-purple');
        this.showOriginalBtn.classList.remove('bg-gray-500');
        this.showSegmentedBtn.classList.add('bg-gray-500');
        this.showSegmentedBtn.classList.remove('bg-brand-purple');
    }

    showSegmentedFrame() {
        this.segmentationOverlay.style.display = 'block';
        this.showSegmentedBtn.classList.add('bg-brand-purple');
        this.showSegmentedBtn.classList.remove('bg-gray-500');
        this.showOriginalBtn.classList.add('bg-gray-500');
        this.showOriginalBtn.classList.remove('bg-brand-purple');
    }

    setLoadingState(loading) {
        if (loading) {
            this.loadingState.classList.remove('hidden');
            this.resultsSection.classList.add('hidden');
            this.startSegmentationBtn.disabled = true;
            this.segmentText.textContent = 'Analyzing...';
            this.segmentSpinner.classList.remove('hidden');
        } else {
            this.loadingState.classList.add('hidden');
            this.startSegmentationBtn.disabled = false;
            this.segmentText.textContent = 'Start Area Detection';
            this.segmentSpinner.classList.add('hidden');
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorState.classList.remove('hidden');
    }

    hideError() {
        this.errorState.classList.add('hidden');
    }

    navigateToFaceRecognition() {
        // Navigate back to face recognition with current parameters
        const params = new URLSearchParams({
            video_path: this.currentVideoPath || '',
            user_id: this.currentUserId || '',
            service_url: this.apiBaseUrl
        });
        window.location.href = `../face-recognition/index.html?${params.toString()}`;
    }

    navigateToPermissions() {
        // Navigate to permissions page with current data
        const params = new URLSearchParams({
            video_path: this.currentVideoPath || '',
            user_id: this.currentUserId || '',
            service_url: this.apiBaseUrl,
            verified_areas: this.verifiedAreas.size,
            total_areas: this.areasData.length
        });
        
        window.location.href = `../permissions/index.html?${params.toString()}`;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AreaSegmentationUI();
});