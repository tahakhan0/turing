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

        // Frame visualization elements
        this.frameImage = document.getElementById('frame-image');
        this.segmentationOverlay = document.getElementById('segmentation-overlay');
        this.showOriginalBtn = document.getElementById('show-original');
        this.showSegmentedBtn = document.getElementById('show-segmented');

        // Areas grid elements
        this.areasGrid = document.getElementById('areas-grid');
        this.filterAreaType = document.getElementById('filter-area-type');
        this.filterVerification = document.getElementById('filter-verification');
        this.verifyAllBtn = document.getElementById('verify-all');

        // Modal elements
        this.modal = document.getElementById('verification-modal');
        this.modalTitle = document.getElementById('modal-title');
        this.modalAreaType = document.getElementById('modal-area-type');
        this.modalConfidence = document.getElementById('modal-confidence');
        this.modalDimensions = document.getElementById('modal-dimensions');
        this.modalAreaSize = document.getElementById('modal-area-size');
        this.modalAreaImage = document.getElementById('modal-area-image');
        this.modalOverlayCanvas = document.getElementById('modal-overlay-canvas');
        this.verifyAreaBtn = document.getElementById('verify-area');
        this.rejectAreaBtn = document.getElementById('reject-area');
        this.closeModalBtn = document.getElementById('close-modal');
        
        // Auto-processing notice
        this.autoProcessingNotice = document.getElementById('auto-processing-notice');
    }

    attachEventListeners() {
        // Navigation
        this.backToFacesBtn.addEventListener('click', () => this.navigateToFaceRecognition());
        this.continueBtn.addEventListener('click', () => this.navigateToPermissions());

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

        // Visualization controls
        this.showOriginalBtn.addEventListener('click', () => this.showOriginalFrame());
        this.showSegmentedBtn.addEventListener('click', () => this.showSegmentedFrame());

        // Verification
        this.verifyAllBtn.addEventListener('click', () => this.verifyAllAreas());

        // Filters
        this.filterAreaType.addEventListener('change', () => this.applyFilters());
        this.filterVerification.addEventListener('change', () => this.applyFilters());

        // Modal
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        this.verifyAreaBtn.addEventListener('click', () => this.verifyCurrentArea(true));
        this.rejectAreaBtn.addEventListener('click', () => this.verifyCurrentArea(false));

        // Close modal on outside click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeModal();
            }
        });
    }

    loadDataFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        this.currentVideoPath = urlParams.get('video_path');
        this.currentUserId = urlParams.get('user_id');
        const facesCount = urlParams.get('faces_count') || '0';

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

        // Auto-start segmentation if coming from face recognition
        if (this.currentUserId && facesCount && parseInt(facesCount) > 0) {
            // Delay to allow UI to initialize
            setTimeout(() => {
                this.autoStartSegmentation();
            }, 1000);
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
        
        // Start the segmentation process
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

        try {
            // First, check if there are already segmented results from extracted frames
            const existingDataResponse = await fetch(`${this.apiBaseUrl}/segmentation/user/${this.currentUserId}`);
            
            if (existingDataResponse.ok) {
                const existingData = await existingDataResponse.json();
                if (existingData.segmentation_data && existingData.segmentation_data.segments && existingData.segmentation_data.segments.length > 0) {
                    // Use existing segmentation data
                    console.log('Found existing segmentation data, displaying results...');
                    this.segmentationData = existingData.segmentation_data;
                    this.areasData = this.segmentationData.segments || [];
                    
                    // Show loading message for existing data
                    this.segmentText.textContent = 'Loading existing segmentation results...';
                    this.startSegmentationBtn.classList.add('bg-blue-600');
                    this.startSegmentationBtn.classList.remove('bg-brand-purple');
                    
                    this.displayResults(null, true); // Show existing results
                    return;
                }
            }

            // If no existing data, try to segment extracted frames
            const segmentFramesResponse = await fetch(`${this.apiBaseUrl}/segmentation/segment/frames/${this.currentUserId}`, {
                method: 'POST'
            });

            if (segmentFramesResponse.ok) {
                const frameSegmentData = await segmentFramesResponse.json();
                
                if (frameSegmentData.status === 'success' && frameSegmentData.total_segments_found > 0) {
                    // Load the newly created segmentation data
                    const newDataResponse = await fetch(`${this.apiBaseUrl}/segmentation/user/${this.currentUserId}`);
                    if (newDataResponse.ok) {
                        const newData = await newDataResponse.json();
                        this.segmentationData = newData.segmentation_data;
                        this.areasData = this.segmentationData.segments || [];
                        this.displayResults(null, true);
                        return;
                    }
                }
            }

            // Fallback to video/image segmentation if available
            if (this.currentVideoPath) {
                console.log('No extracted frames available, trying direct video segmentation');
                
                // Extract frame from video first
                const frameNumber = parseInt(this.frameNumberInput.value) || 1;
                const extractResponse = await fetch(`${this.apiBaseUrl}/face-recognition/extract-frame`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        video_path: this.currentVideoPath,
                        frame_number: frameNumber,
                        user_id: this.currentUserId
                    })
                });

                if (!extractResponse.ok) {
                    throw new Error(`Failed to extract frame: ${extractResponse.statusText}`);
                }

                const frameData = await extractResponse.json();
                const imagePath = frameData.image_path;

                // Now segment the extracted frame using Replicate
                const segmentResponse = await fetch(`${this.apiBaseUrl}/segmentation/segment/replicate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image_path: imagePath,
                        user_id: this.currentUserId,
                        box_threshold: parseFloat(this.boxThresholdSlider.value),
                        text_threshold: parseFloat(this.textThresholdSlider.value)
                    })
                });

                if (!segmentResponse.ok) {
                    throw new Error(`Segmentation failed: ${segmentResponse.statusText}`);
                }

                this.segmentationData = await segmentResponse.json();
                console.log('Segmentation response:', this.segmentationData);
                
                if (this.segmentationData.status === 'error') {
                    throw new Error(this.segmentationData.error);
                }

                this.areasData = this.segmentationData.segments || [];
                this.displayResults(null, false); // Pass null so it uses segmentationData.image_path
            } else {
                throw new Error('No video path provided and no extracted frames found. Please complete face recognition first.');
            }
            
        } catch (error) {
            console.error('Segmentation error:', error);
            this.showError(error.message);
        } finally {
            this.setLoadingState(false);
        }
    }

    displayResults(imagePath, isFromExtractedFrames = false) {
        this.setupSection.classList.add('hidden');
        this.resultsSection.classList.remove('hidden');

        // Update summary
        this.updateSummary();

        // Display frame - handle different image sources
        if (isFromExtractedFrames && this.segmentationData.frame_details && this.segmentationData.frame_details.length > 0) {
            // Use the first frame that has a visualization
            const frameWithViz = this.segmentationData.frame_details.find(f => f.visualization_url);
            if (frameWithViz && frameWithViz.visualization_url) {
                this.frameImage.src = frameWithViz.visualization_url;
            } else {
                // Fallback to first available frame
                const firstFrame = this.areasData.find(area => area.source_frame_path);
                if (firstFrame && firstFrame.source_frame_path) {
                    this.frameImage.src = `${this.apiBaseUrl}/static/extracted_frames/${firstFrame.source_frame_path.split('/').pop()}`;
                }
            }
        } else if (imagePath) {
            // Single image segmentation - check if it's already a full URL path
            if (imagePath.startsWith('/static/')) {
                this.frameImage.src = imagePath;
            } else {
                this.frameImage.src = `${this.apiBaseUrl}/static/extracted_frames/${imagePath.split('/').pop()}`;
            }
        } else if (this.segmentationData.image_path) {
            // Use image_path from segmentation response (could be visualization)
            console.log('Using image_path from response:', this.segmentationData.image_path);
            this.frameImage.src = this.segmentationData.image_path;
        } else {
            // Try to use visualization URL from segmentation data
            if (this.segmentationData.visualization_url) {
                console.log('Using visualization URL:', this.segmentationData.visualization_url);
                this.frameImage.src = this.segmentationData.visualization_url;
            } else {
                console.log('No visualization URL found in segmentation data:', this.segmentationData);
            }
        }

        // Image will display directly without overlays

        // Display areas grid
        this.displayAreasGrid();

        // Enable continue button if areas are detected
        if (this.areasData.length > 0) {
            this.continueBtn.disabled = false;
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

    displayAreasGrid() {
        this.areasGrid.innerHTML = '';

        const filteredAreas = this.getFilteredAreas();

        filteredAreas.forEach((area, index) => {
            const areaCard = this.createAreaCard(area, index);
            this.areasGrid.appendChild(areaCard);
        });
    }

    createAreaCard(area, index) {
        const colors = [
            'border-red-200 bg-red-50', 'border-orange-200 bg-orange-50', 
            'border-yellow-200 bg-yellow-50', 'border-green-200 bg-green-50',
            'border-cyan-200 bg-cyan-50', 'border-blue-200 bg-blue-50', 
            'border-purple-200 bg-purple-50', 'border-pink-200 bg-pink-50'
        ];

        const isVerified = this.verifiedAreas.has(area.area_id);
        const cardColor = colors[index % colors.length];

        const card = document.createElement('div');
        card.className = `area-card border-2 ${cardColor} rounded-lg p-4 cursor-pointer transition-all ${isVerified ? 'ring-2 ring-green-500' : ''}`;
        
        const areaType = area.area_type || area.label || 'unknown';
        const confidence = area.confidence || 0;
        
        card.innerHTML = `
            <div class="flex justify-between items-start mb-3">
                <div class="flex-1">
                    <h3 class="font-medium text-gray-900 capitalize mb-1">${areaType.replace('_', ' ')}</h3>
                    <p class="text-sm text-gray-600">Confidence: ${(confidence * 100).toFixed(1)}%</p>
                </div>
                ${isVerified ? '<div class="text-green-600"><svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg></div>' : ''}
            </div>
            
            <div class="space-y-1 text-sm text-gray-600">
                <div>Type: ${areaType}</div>
                <div>Source: ${area.source_frame || 'N/A'}</div>
                <div>Detection: ${area.source || 'N/A'}</div>
            </div>
            
            <div class="mt-4 flex space-x-2">
                <button class="verify-btn flex-1 text-xs px-3 py-2 rounded ${isVerified ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'} transition-colors">
                    ${isVerified ? '✓ Verified' : 'Verify'}
                </button>
                <button class="details-btn text-xs px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors">
                    Details
                </button>
            </div>
        `;

        // Add event listeners
        const verifyBtn = card.querySelector('.verify-btn');
        const detailsBtn = card.querySelector('.details-btn');

        verifyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (!isVerified) {
                this.openVerificationModal(area);
            }
        });

        detailsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.openVerificationModal(area);
        });

        return card;
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
        this.displayAreasGrid();
    }

    openVerificationModal(area) {
        this.currentAreaForVerification = area;
        
        const areaType = area.area_type || area.label || 'unknown';
        const confidence = area.confidence || 0;
        
        this.modalAreaType.textContent = areaType.replace('_', ' ');
        this.modalConfidence.textContent = (confidence * 100).toFixed(1) + '%';
        this.modalDimensions.textContent = area.bbox ? 
            `${area.bbox.x2 - area.bbox.x1}px × ${area.bbox.y2 - area.bbox.y1}px` : 
            `${area.dimensions?.width?.toFixed(1) || 'N/A'}m × ${area.dimensions?.height?.toFixed(1) || 'N/A'}m`;
        this.modalAreaSize.textContent = area.dimensions?.area?.toFixed(1) + 'm²' || 'N/A';
        
        // Set the same frame image
        this.modalAreaImage.src = this.frameImage.src;
        this.modalAreaImage.onload = () => {
            this.drawModalOverlay(area);
        };

        this.modal.classList.remove('hidden');
    }

    drawModalOverlay(area) {
        const canvas = this.modalOverlayCanvas;
        const ctx = canvas.getContext('2d');
        const img = this.modalAreaImage;

        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        canvas.style.width = img.clientWidth + 'px';
        canvas.style.height = img.clientHeight + 'px';

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const scaleX = canvas.width / img.naturalWidth;
        const scaleY = canvas.height / img.naturalHeight;

        ctx.strokeStyle = '#ef4444';
        ctx.fillStyle = '#ef444440';
        ctx.lineWidth = 3;

        if (area.bbox) {
            // Draw bounding box for Replicate API results
            const x1 = area.bbox.x1 * scaleX;
            const y1 = area.bbox.y1 * scaleY;
            const x2 = area.bbox.x2 * scaleX;
            const y2 = area.bbox.y2 * scaleY;

            ctx.beginPath();
            ctx.rect(x1, y1, x2 - x1, y2 - y1);
            ctx.fill();
            ctx.stroke();
        } else if (area.polygon && area.polygon.length > 0) {
            // Draw polygon for other API results
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
        }
    }

    closeModal() {
        this.modal.classList.add('hidden');
        this.currentAreaForVerification = null;
    }

    async verifyCurrentArea(approved) {
        if (!this.currentAreaForVerification) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/segmentation/segment/verify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    area_id: this.currentAreaForVerification.area_id,
                    user_id: this.currentUserId,
                    approved: approved
                })
            });

            if (!response.ok) {
                throw new Error(`Verification failed: ${response.statusText}`);
            }

            const result = await response.json();

            if (approved) {
                this.verifiedAreas.add(this.currentAreaForVerification.area_id);
            } else {
                // Remove from areas data if rejected
                this.areasData = this.areasData.filter(area => area.area_id !== this.currentAreaForVerification.area_id);
            }

            this.updateSummary();
            this.displayAreasGrid();
            this.closeModal();

        } catch (error) {
            console.error('Verification error:', error);
            this.showError(error.message);
        }
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
        this.displayAreasGrid();
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