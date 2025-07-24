// Turing Face Recognition Web UI - Main Application JavaScript

class FaceRecognitionUI {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentAnalysisData = null;
        this.faceData = [];
        this.frameGroups = new Map(); // Group detections by frame for consolidated labeling
        this.labeledFaces = new Set();
        this.namesSuggestions = new Set();
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkServiceConnection();
    }

    initializeElements() {
        // Form elements
        this.userIdInput = document.getElementById('user-id');
        this.videoPathInput = document.getElementById('video-path');
        this.folderPathInput = document.getElementById('folder-path');
        this.serviceUrlInput = document.getElementById('service-url');
        this.analyzeBtn = document.getElementById('analyze-video');
        this.analyzeText = document.getElementById('analyze-text');
        this.analyzeSpinner = document.getElementById('analyze-spinner');
        this.continueToSegmentationBtn = document.getElementById('continue-to-segmentation');
        
        // Analysis type controls
        this.singleVideoOption = document.getElementById('single-video-option');
        this.folderOption = document.getElementById('folder-option');
        this.recognitionOption = document.getElementById('recognition-option');
        this.videoPathSection = document.getElementById('video-path-section');
        this.folderPathSection = document.getElementById('folder-path-section');

        // Results elements
        this.resultsSection = document.getElementById('results-section');
        this.loadingState = document.getElementById('loading-state');
        this.errorState = document.getElementById('error-state');
        this.errorMessage = document.getElementById('error-message');
        this.facesGrid = document.getElementById('faces-grid');

        // Summary elements
        this.totalFrames = document.getElementById('total-frames');
        this.analyzedFrames = document.getElementById('analyzed-frames');
        this.totalFaces = document.getElementById('total-faces');
        this.labeledFacesCount = document.getElementById('labeled-faces');

        // Modal elements
        this.modal = document.getElementById('labeling-modal');
        this.modalTitle = document.getElementById('modal-title');
        this.modalFaceImage = document.getElementById('modal-face-image');
        this.modalFrameNumber = document.getElementById('modal-frame-number');
        
        // Single labeling elements
        this.singleLabelingForm = document.getElementById('single-labeling-form');
        this.personNameInput = document.getElementById('person-name');
        this.nameSuggestions = document.getElementById('name-suggestions');
        this.confirmBtn = document.getElementById('confirm-face');
        this.skipBtn = document.getElementById('skip-face');
        this.rejectBtn = document.getElementById('reject-face');
        
        // Bulk labeling elements
        this.bulkLabelingForm = document.getElementById('bulk-labeling-form');
        this.bulkPersonInputs = document.getElementById('bulk-person-inputs');
        this.bulkNameSuggestions = document.getElementById('bulk-name-suggestions');
        this.confirmBulkBtn = document.getElementById('confirm-bulk');
        this.skipBulkBtn = document.getElementById('skip-bulk');
        
        this.closeModalBtn = document.getElementById('close-modal');

        // Filter elements
        this.filterStatus = document.getElementById('filter-status');
        this.searchNames = document.getElementById('search-names');
        this.saveAllBtn = document.getElementById('save-all');

        // Connection status
        this.connectionStatus = document.getElementById('connection-status');

        this.currentFaceData = null;
    }

    attachEventListeners() {
        // Service URL change
        this.serviceUrlInput.addEventListener('change', () => {
            this.apiBaseUrl = this.serviceUrlInput.value;
            this.checkServiceConnection();
        });

        // Analysis type change
        this.singleVideoOption.addEventListener('change', () => this.toggleAnalysisType());
        this.folderOption.addEventListener('change', () => this.toggleAnalysisType());
        this.recognitionOption.addEventListener('change', () => this.toggleAnalysisType());

        // Analyze video
        this.analyzeBtn.addEventListener('click', () => this.analyzeVideo());
        
        // Continue to segmentation
        this.continueToSegmentationBtn.addEventListener('click', () => this.navigateToSegmentation());

        // Modal controls
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        
        // Single labeling controls
        this.confirmBtn.addEventListener('click', () => this.confirmFace());
        this.skipBtn.addEventListener('click', () => this.skipFace());
        this.rejectBtn.addEventListener('click', () => this.rejectFace());
        
        // Bulk labeling controls
        this.confirmBulkBtn.addEventListener('click', () => this.confirmBulkLabels());
        this.skipBulkBtn.addEventListener('click', () => this.skipBulkLabels());

        // Filter controls
        this.filterStatus.addEventListener('change', () => this.filterFaces());
        this.searchNames.addEventListener('input', () => this.filterFaces());

        // Save all
        this.saveAllBtn.addEventListener('click', () => this.saveAllLabels());

        // Close modal on outside click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeModal();
            }
        });

        // Enter key support for person name
        this.personNameInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.confirmFace();
            }
        });
    }

    async checkServiceConnection() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
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
        const statusDot = this.connectionStatus.querySelector('div');
        const statusText = this.connectionStatus.querySelector('span');
        
        if (connected) {
            statusDot.className = 'w-2 h-2 bg-green-400 rounded-full';
            statusText.textContent = 'Connected';
            statusText.className = 'text-sm text-gray-600';
        } else {
            statusDot.className = 'w-2 h-2 bg-red-400 rounded-full';
            statusText.textContent = 'Disconnected';
            statusText.className = 'text-sm text-gray-600';
        }
    }

    toggleAnalysisType() {
        const isFolderMode = this.folderOption.checked;
        const isRecognitionMode = this.recognitionOption.checked;
        
        if (isFolderMode) {
            this.videoPathSection.classList.add('hidden');
            this.folderPathSection.classList.remove('hidden');
        } else if (isRecognitionMode) {
            // Recognition mode can work with both single video and folder
            // For now, show both options but we'll handle the logic in analyzeVideo
            this.videoPathSection.classList.remove('hidden');
            this.folderPathSection.classList.remove('hidden');
        } else {
            // Single video mode
            this.videoPathSection.classList.remove('hidden');
            this.folderPathSection.classList.add('hidden');
        }
    }

    async analyzeVideo() {
        const userId = this.userIdInput.value.trim();
        const isFolderMode = this.folderOption.checked;
        const isRecognitionMode = this.recognitionOption.checked;
        
        let payload = { user_id: userId };
        let pathValue = '';
        let endpoint = '/face-recognition/enrollment';
        
        if (isRecognitionMode) {
            endpoint = '/face-recognition/analyze';
        }
        
        if (isFolderMode) {
            // Folder mode - use folder path
            pathValue = this.folderPathInput.value.trim();
            if (!userId || !pathValue) {
                this.showError('Please enter both User ID and Folder Path');
                return;
            }
            payload.folder_path = pathValue;
        } else if (isRecognitionMode) {
            // Recognition mode - check which input has a value
            const videoPath = this.videoPathInput.value.trim();
            const folderPath = this.folderPathInput.value.trim();
            
            if (!userId) {
                this.showError('Please enter User ID');
                return;
            }
            
            if (videoPath && folderPath) {
                this.showError('Please enter either Video Path OR Folder Path, not both');
                return;
            }
            
            if (!videoPath && !folderPath) {
                this.showError('Please enter either Video Path or Folder Path');
                return;
            }
            
            if (videoPath) {
                payload.video_path = videoPath;
                pathValue = videoPath;
            } else {
                payload.folder_path = folderPath;
                pathValue = folderPath;
            }
        } else {
            // Single video mode - use video path
            pathValue = this.videoPathInput.value.trim();
            if (!userId || !pathValue) {
                this.showError('Please enter both User ID and Video Path');
                return;
            }
            payload.video_path = pathValue;
        }

        this.setAnalyzing(true);
        this.hideError();
        this.hideResults();
        this.showLoading();

        try {
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Analysis response:', data);
            
            // The API returns VideoAnalysis or FaceRecognitionAnalysis directly, no wrapper
            this.currentAnalysisData = { ...data, user_id: userId };
            this.processFaceData(data);
            this.updateSummary(data);
            this.renderFaceGrid();
            this.showResults();

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(`Analysis failed: ${error.message}`);
        } finally {
            this.setAnalyzing(false);
            this.hideLoading();
        }
    }

    processFaceData(analysisData) {
        this.faceData = [];
        this.frameGroups = new Map(); // Group detections by frame for consolidated labeling
        let faceId = 0;

        console.log('Processing face data from:', analysisData);
        
        // Check if this is a recognition analysis (has detections with frame-by-frame data)
        if (analysisData.detections !== undefined && analysisData.total_frames !== undefined) {
            // New face recognition analysis format
            console.log('Processing face recognition analysis with detections:', analysisData);
            
            analysisData.detections.forEach(frameData => {
                // Only process frames that have actual face detections
                if (frameData.detections && frameData.detections.length > 0) {
                    const frameKey = frameData.frame_number;
                    if (!this.frameGroups.has(frameKey)) {
                        this.frameGroups.set(frameKey, {
                            frameNumber: frameData.frame_number,
                            detections: [],
                            visualizationUrl: frameData.visualization_url,
                            videoPath: analysisData.video_path,
                            videoName: analysisData.video_path ? analysisData.video_path.split('/').pop() : 'Unknown'
                        });
                    }
                    
                    frameData.detections.forEach(detection => {
                        const isRecognized = detection.recognition_status === 'recognized';
                        
                        const faceData = {
                            id: faceId++,
                            frameNumber: frameData.frame_number,
                            timestamp: frameData.timestamp,
                            bbox: detection.bbox,
                            confidence: 1.0, // Face recognition doesn't provide confidence scores
                            visualizationUrl: frameData.visualization_url,
                            videoName: analysisData.video_path || 'Unknown',
                            videoPath: analysisData.video_path,
                            labeled: isRecognized,
                            personName: detection.person_name || detection.name || '',
                            status: isRecognized ? 'recognized' : 'unrecognized',
                            recognitionStatus: detection.recognition_status,
                            faceEncoding: detection.face_encoding, // Store for potential saving
                            detectionType: 'face',
                            personId: detection.person_id || 0
                        };
                        
                        this.faceData.push(faceData);
                        this.frameGroups.get(frameKey).detections.push(faceData);
                    });
                } else {
                    console.log(`Skipping frame ${frameData.frame_number} - no face detections found`);
                }
            });
        } else if (analysisData.unique_persons !== undefined) {
            // Directory analysis format
            console.log('Processing directory analysis with unique persons:', analysisData.unique_persons);
            
            analysisData.detections.forEach(detection => {
                console.log('Processing unique person detection:', detection);
                
                const frameKey = detection.frame_number;
                if (!this.frameGroups.has(frameKey)) {
                    this.frameGroups.set(frameKey, {
                        frameNumber: detection.frame_number,
                        detections: [],
                        visualizationUrl: detection.visualization_url,
                        videoPath: detection.video_path,
                        videoName: detection.video_name || 'Unknown'
                    });
                }
                
                const faceData = {
                    id: faceId++,
                    frameNumber: detection.frame_number,
                    timestamp: detection.timestamp,
                    bbox: detection.bbox,
                    confidence: detection.confidence,
                    visualizationUrl: detection.visualization_url,
                    videoName: detection.video_name || 'Unknown',
                    videoPath: detection.video_path,
                    labeled: false,
                    personName: '',
                    status: 'unlabeled',
                    detectionType: 'person',
                    personId: 0
                };
                
                this.faceData.push(faceData);
                this.frameGroups.get(frameKey).detections.push(faceData);
            });
        } else if (analysisData.analysis !== undefined) {
            // Single video analysis format (VideoAnalysis)
            console.log('Processing single video analysis, analysis array:', analysisData.analysis);
            
            analysisData.analysis.forEach(frameAnalysis => {
                console.log('Processing frame analysis:', frameAnalysis);
                
                // Include both person and face detections
                const allDetections = frameAnalysis.detections.filter(d => 
                    d.class_name === 'person' || d.class_name === 'face'
                );
                console.log('Person and face detections in this frame:', allDetections);
                
                // Only create frame group if there are actual detections
                if (allDetections.length > 0) {
                    const frameKey = frameAnalysis.frame_number;
                    if (!this.frameGroups.has(frameKey)) {
                        this.frameGroups.set(frameKey, {
                            frameNumber: frameAnalysis.frame_number,
                            detections: [],
                            visualizationUrl: frameAnalysis.visualization_url,
                            videoPath: analysisData.video_path,
                            videoName: analysisData.video_path ? analysisData.video_path.split('/').pop() : 'Unknown'
                        });
                    }
                    
                    allDetections.forEach(detection => {
                        // Check if this detection is already labeled/recognized
                        const isLabeled = detection.person_name && detection.person_name.trim() !== '';
                        
                        const faceData = {
                            id: faceId++,
                            frameNumber: frameAnalysis.frame_number,
                            timestamp: null,
                            bbox: detection.bbox,
                            confidence: detection.confidence,
                            visualizationUrl: frameAnalysis.visualization_url,
                            videoName: analysisData.video_path || 'Unknown',
                            videoPath: analysisData.video_path,
                            labeled: isLabeled,
                            personName: detection.person_name || '',
                            status: isLabeled ? 'labeled' : 'unlabeled',
                            detectionType: detection.detection_type || 'person', // Store detection type
                            personId: detection.person_id || 0 // Store person ID
                        };
                        
                        this.faceData.push(faceData);
                        this.frameGroups.get(frameKey).detections.push(faceData);
                    });
                } else {
                    console.log(`Skipping frame ${frameAnalysis.frame_number} - no person or face detections found`);
                }
            });
        }

        console.log(`Processed ${this.faceData.length} faces from analysis`);
        console.log('Frame groups:', this.frameGroups);
        console.log('Face data:', this.faceData);
    }

    updateSummary(data) {
        // Handle different analysis formats
        if (data.detections !== undefined && data.total_frames !== undefined) {
            // New face recognition analysis format
            this.totalFrames.textContent = data.total_frames;
            this.analyzedFrames.textContent = data.processed_frames;
            this.totalFaces.textContent = data.recognized_faces + data.unrecognized_faces;
            this.labeledFacesCount.textContent = data.recognized_faces;
        } else if (data.unique_persons !== undefined) {
            // Directory analysis format
            this.totalFrames.textContent = data.total_videos || 0;
            this.analyzedFrames.textContent = data.processed_videos || 0;
            this.totalFaces.textContent = data.unique_persons || 0;
            this.updateLabeledCount();
        } else if (data.analysis !== undefined) {
            // Single video analysis format (VideoAnalysis)
            this.totalFrames.textContent = data.analysis ? data.analysis.length : 0;
            this.analyzedFrames.textContent = data.analysis ? data.analysis.length : 0;
            this.totalFaces.textContent = this.faceData.length;
            this.updateLabeledCount();
        } else {
            // Unknown format
            console.warn('Unknown analysis data format:', data);
            this.totalFrames.textContent = 0;
            this.analyzedFrames.textContent = 0;
            this.totalFaces.textContent = this.faceData.length;
            this.updateLabeledCount();
        }
    }

    updateLabeledCount() {
        const labeled = this.faceData.filter(face => face.labeled).length;
        this.labeledFacesCount.textContent = labeled;
        
        // Enable continue button if we have analysis data and some faces are labeled
        if (this.currentAnalysisData && labeled > 0) {
            this.continueToSegmentationBtn.disabled = false;
        }
    }

    renderFaceGrid() {
        const filteredFrameGroups = this.getFilteredFrameGroups();
        
        console.log('Rendering face grid with', filteredFrameGroups.length, 'frame groups');
        console.log('Filtered frame groups:', filteredFrameGroups);
        
        this.facesGrid.innerHTML = '';

        if (filteredFrameGroups.length === 0) {
            console.log('No frame groups to display');
            this.facesGrid.innerHTML = `
                <div class="col-span-full text-center py-8 text-gray-500">
                    No frames match the current filter
                </div>
            `;
            return;
        }

        filteredFrameGroups.forEach(frameGroup => {
            console.log('Creating card for frame group:', frameGroup);
            const faceCard = this.createFaceCard(frameGroup);
            this.facesGrid.appendChild(faceCard);
        });
    }

    createFaceCard(frameGroup) {
        const card = document.createElement('div');
        const hasLabeled = frameGroup.detections.some(d => d.labeled);
        const allLabeled = frameGroup.detections.every(d => d.labeled);
        
        const borderColor = allLabeled ? 'border-green-300' : hasLabeled ? 'border-yellow-300' : 'border-gray-200';
        
        card.className = `face-card bg-white border ${borderColor} rounded-lg overflow-hidden cursor-pointer hover:shadow-md transition-all`;
        
        const statusText = allLabeled ? 'All Labeled' : hasLabeled ? 'Partially Labeled' : 'Unlabeled';
        const statusColor = allLabeled ? 'bg-green-100 text-green-700' : hasLabeled ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-700';

        // Count detection types
        const personCount = frameGroup.detections.filter(d => d.detectionType === 'person').length;
        const faceCount = frameGroup.detections.filter(d => d.detectionType === 'face').length;
        
        const detectionSummary = [];
        if (personCount > 0) detectionSummary.push(`${personCount} Person${personCount > 1 ? 's' : ''}`);
        if (faceCount > 0) detectionSummary.push(`${faceCount} Face${faceCount > 1 ? 's' : ''}`);

        card.innerHTML = `
            <div class="aspect-square bg-gray-50 flex items-center justify-center relative">
                ${frameGroup.visualizationUrl ? 
                    `<img src="${frameGroup.visualizationUrl}" alt="Frame detections" class="w-full h-full object-cover">` :
                    `<div class="text-gray-400">No Image</div>`
                }
                <div class="absolute top-2 left-2">
                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-white bg-opacity-90 text-gray-800">
                        ${frameGroup.detections.length} Detection${frameGroup.detections.length > 1 ? 's' : ''}
                    </span>
                </div>
            </div>
            <div class="p-4">
                <div class="flex justify-between items-start mb-2">
                    <div class="text-sm text-gray-600">Frame ${frameGroup.frameNumber}</div>
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusColor}">
                        ${statusText}
                    </span>
                </div>
                ${frameGroup.videoName ? `<div class="text-xs text-gray-400 mb-2">${frameGroup.videoName}</div>` : ''}
                <div class="text-sm text-gray-500 mb-2">
                    ${detectionSummary.join(' + ')}
                </div>
                <div class="text-sm text-gray-400">
                    Click to label all detections
                </div>
            </div>
        `;

        card.addEventListener('click', () => this.openConsolidatedLabelingModal(frameGroup));
        return card;
    }

    getFilteredFrameGroups() {
        let filteredGroups = Array.from(this.frameGroups.values());

        // Filter by status
        const statusFilter = this.filterStatus.value;
        if (statusFilter !== 'all') {
            filteredGroups = filteredGroups.filter(frameGroup => {
                const hasLabeled = frameGroup.detections.some(d => d.labeled);
                const allLabeled = frameGroup.detections.every(d => d.labeled);
                
                if (statusFilter === 'labeled') {
                    return allLabeled;
                } else if (statusFilter === 'unlabeled') {
                    return !hasLabeled;
                }
                return true;
            });
        }

        // Filter by name search
        const searchTerm = this.searchNames.value.toLowerCase().trim();
        if (searchTerm) {
            filteredGroups = filteredGroups.filter(frameGroup => 
                frameGroup.detections.some(d => 
                    (d.personName || '').toLowerCase().includes(searchTerm)
                )
            );
        }

        return filteredGroups.sort((a, b) => a.frameNumber - b.frameNumber);
    }

    filterFaces() {
        this.renderFaceGrid();
    }

    async openConsolidatedLabelingModal(frameGroup) {
        this.currentFrameGroup = frameGroup;
        console.log('Opening consolidated labeling modal for frame group:', frameGroup);
        
        // Set modal title
        this.modalTitle.textContent = frameGroup.detections.length > 1 ? 'Label All Detections' : 'Label Detection';
        
        // Set frame image
        if (frameGroup.visualizationUrl) {
            this.modalFaceImage.src = frameGroup.visualizationUrl;
        } else {
            this.modalFaceImage.src = '';
        }
        
        // Set frame number and detection count
        this.modalFrameNumber.textContent = `Frame ${frameGroup.frameNumber} (${frameGroup.detections.length} detection${frameGroup.detections.length > 1 ? 's' : ''})`;
        
        // Hide single labeling form and show consolidated interface
        this.hideSingleLabelingControls();
        this.showConsolidatedLabelingControls(frameGroup);
        
        // Show modal
        this.modal.classList.remove('hidden');
    }

    showConsolidatedLabelingControls(frameGroup) {
        // Use the bulk labeling form but modify it for consolidated interface
        this.bulkLabelingForm.classList.remove('hidden');
        this.createConsolidatedPersonInputs(frameGroup);
        this.updateBulkNameSuggestions();
    }

    createConsolidatedPersonInputs(frameGroup) {
        this.bulkPersonInputs.innerHTML = '';
        
        // Color names for display (matching the colors in the backend)
        const colorNames = [
            'Green', 'Blue', 'Red', 'Cyan', 'Magenta', 'Yellow',
            'Purple', 'Orange', 'Light Blue', 'Lime', 'Deep Pink', 'Dark Turquoise'
        ];
        
        // Create header
        const headerDiv = document.createElement('div');
        headerDiv.className = 'bg-blue-50 border border-blue-200 rounded-md p-4 mb-4';
        headerDiv.innerHTML = `
            <h4 class="text-sm font-medium text-blue-800 mb-2">Label All Detections</h4>
            <p class="text-sm text-blue-700">Each detection has a unique identifier. Label them individually below:</p>
        `;
        this.bulkPersonInputs.appendChild(headerDiv);
        
        frameGroup.detections.forEach((detection, index) => {
            const colorName = detection.detectionType === 'person' && detection.personId !== undefined ? 
                colorNames[detection.personId] || `Color ${detection.personId}` : null;
            
            const personDiv = document.createElement('div');
            personDiv.className = 'border border-gray-200 rounded-md p-4 bg-gray-50 mb-3';
            
            // Detection type badge with color coding
            const badgeColor = detection.detectionType === 'face' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800';
            const detectionTypeBadge = `<span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${badgeColor}">
                ${detection.detectionType === 'face' ? 'Face' : 'Person'}
            </span>`;
            
            const colorInfo = colorName ? `<span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 ml-2">
                ${colorName} Box
            </span>` : '';
            
            personDiv.innerHTML = `
                <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center">
                        ${detectionTypeBadge}
                        ${colorInfo}
                    </div>
                    <div class="text-xs text-gray-500">
                        Confidence: ${(detection.confidence * 100).toFixed(1)}%
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">
                        Person Name for ${detection.detectionType === 'face' ? 'Face' : 'Person'}${colorName ? ` (${colorName} Box)` : ''}
                    </label>
                    <input 
                        type="text" 
                        class="consolidated-person-input w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-brand-blue focus:border-brand-blue" 
                        placeholder="Enter person's name" 
                        value="${detection.personName || ''}" 
                        data-detection-index="${index}"
                        data-person-id="${detection.personId || 0}"
                        data-detection-type="${detection.detectionType}"
                        data-bbox-x1="${detection.bbox.x1}"
                        data-bbox-y1="${detection.bbox.y1}"
                        data-bbox-x2="${detection.bbox.x2}"
                        data-bbox-y2="${detection.bbox.y2}"
                    >
                </div>
            `;
            
            this.bulkPersonInputs.appendChild(personDiv);
        });
    }

    async openLabelingModal(face) {
        // This is kept for backward compatibility with old code paths
        // but new code should use openConsolidatedLabelingModal
        this.currentFaceData = face;
        
        // Check if we need to fetch frame detection info for bulk labeling
        await this.setupLabelingModal(face);
        
        // Show modal
        this.modal.classList.remove('hidden');
    }

    async setupLabelingModal(face) {
        try {
            // Check if this frame has multiple people by fetching detection info
            const frameDetections = await this.getFrameDetections(face);
            
            if (frameDetections && frameDetections.total_persons > 1) {
                // Multiple people detected - show bulk labeling interface
                this.setupBulkLabelingModal(face, frameDetections);
            } else {
                // Single person - show regular labeling interface
                this.setupSingleLabelingModal(face);
            }
        } catch (error) {
            console.error('Error setting up labeling modal:', error);
            // Fallback to single person labeling
            this.setupSingleLabelingModal(face);
        }
    }

    async getFrameDetections(face) {
        try {
            const userId = this.currentAnalysisData.user_id;
            const videoPath = face.videoPath || this.currentAnalysisData.video_path;
            const frameNumber = face.frameNumber;
            
            const response = await fetch(
                `${this.apiBaseUrl}/face-recognition/frame-detections/${userId}/${frameNumber}?video_path=${encodeURIComponent(videoPath)}`
            );
            
            if (!response.ok) {
                throw new Error(`Failed to fetch frame detections: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching frame detections:', error);
            return null;
        }
    }

    setupSingleLabelingModal(face) {
        // Set face image
        if (face.visualizationUrl) {
            this.modalFaceImage.src = face.visualizationUrl;
        } else {
            this.modalFaceImage.src = '';
        }
        
        // Set frame number with detection type information
        const detectionTypeText = face.detectionType === 'face' ? 'Face Detection' : 'Person Detection';
        const colorNames = [
            'Green', 'Blue', 'Red', 'Cyan', 'Magenta', 'Yellow',
            'Purple', 'Orange', 'Light Blue', 'Lime', 'Deep Pink', 'Dark Turquoise'
        ];
        const colorInfo = face.detectionType === 'person' && face.personId !== undefined ? 
            ` (${colorNames[face.personId] || `Color ${face.personId}`} Box)` : '';
        
        this.modalFrameNumber.textContent = `Frame ${face.frameNumber} - ${detectionTypeText}${colorInfo}`;
        
        // Set current name
        this.personNameInput.value = face.personName;
        
        // Update name suggestions
        this.updateNameSuggestions();
        
        // Hide bulk labeling controls if they exist
        this.hideBulkLabelingControls();
        
        // Show single person controls
        this.showSingleLabelingControls();
        
        this.personNameInput.focus();
    }

    setupBulkLabelingModal(face, frameDetections) {
        // Set face image showing all detections
        if (face.visualizationUrl) {
            this.modalFaceImage.src = face.visualizationUrl;
        }
        
        // Set frame number
        this.modalFrameNumber.textContent = `${face.frameNumber} (${frameDetections.total_persons} people detected)`;
        
        // Hide single person controls
        this.hideSingleLabelingControls();
        
        // Show bulk labeling controls
        this.showBulkLabelingControls(frameDetections);
    }

    closeModal() {
        this.modal.classList.add('hidden');
        this.currentFaceData = null;
        this.currentFrameDetections = null;
        this.currentFrameGroup = null; // Clear consolidated interface data
        this.personNameInput.value = '';
        
        // Clear bulk inputs
        if (this.bulkPersonInputs) {
            this.bulkPersonInputs.innerHTML = '';
        }
    }

    showSingleLabelingControls() {
        this.singleLabelingForm.classList.remove('hidden');
    }

    hideSingleLabelingControls() {
        this.singleLabelingForm.classList.add('hidden');
    }

    showBulkLabelingControls(frameDetections) {
        this.currentFrameDetections = frameDetections;
        this.bulkLabelingForm.classList.remove('hidden');
        this.createBulkPersonInputs(frameDetections);
        this.updateBulkNameSuggestions();
    }

    hideBulkLabelingControls() {
        this.bulkLabelingForm.classList.add('hidden');
        this.currentFrameDetections = null;
    }

    createBulkPersonInputs(frameDetections) {
        this.bulkPersonInputs.innerHTML = '';
        
        // Color names for display (matching the colors in the backend)
        const colorNames = [
            'Green', 'Blue', 'Red', 'Cyan', 'Magenta', 'Yellow',
            'Purple', 'Orange', 'Light Blue', 'Lime', 'Deep Pink', 'Dark Turquoise'
        ];
        
        frameDetections.detections.forEach((detection, index) => {
            const colorName = colorNames[detection.person_id] || `Color ${detection.person_id}`;
            
            const personDiv = document.createElement('div');
            personDiv.className = 'border border-gray-200 rounded-md p-4 bg-gray-50';
            personDiv.innerHTML = `
                <div class="flex items-center justify-between mb-3">
                    <label class="text-sm font-medium text-gray-700">
                        Person ${detection.person_id} (${colorName} Box)
                    </label>
                    <span class="text-xs text-gray-500">
                        Confidence: ${(detection.confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <input 
                    type="text" 
                    id="person-${detection.person_id}-name" 
                    data-person-id="${detection.person_id}"
                    data-bbox='${JSON.stringify(detection.bbox)}'
                    data-detection-type="${detection.detection_type}"
                    placeholder="Enter person's name" 
                    class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-brand-blue focus:border-brand-blue text-sm"
                >
            `;
            
            this.bulkPersonInputs.appendChild(personDiv);
        });
    }

    updateBulkNameSuggestions() {
        // Get unique names from labeled faces for bulk suggestions
        const uniqueNames = [...new Set(
            this.faceData
                .filter(face => face.personName)
                .map(face => face.personName)
        )];

        this.bulkNameSuggestions.innerHTML = '';
        
        uniqueNames.forEach(name => {
            const button = document.createElement('button');
            button.className = 'px-3 py-1 bg-gray-100 text-gray-700 rounded-md text-sm hover:bg-gray-200 border border-gray-300 transition-colors';
            button.textContent = name;
            button.addEventListener('click', () => {
                // Fill the first empty input with this name
                const inputs = this.bulkPersonInputs.querySelectorAll('input[type="text"]');
                for (const input of inputs) {
                    if (!input.value.trim()) {
                        input.value = name;
                        break;
                    }
                }
            });
            this.bulkNameSuggestions.appendChild(button);
        });
    }

    updateNameSuggestions() {
        // Get unique names from labeled faces
        const uniqueNames = [...new Set(
            this.faceData
                .filter(face => face.personName)
                .map(face => face.personName)
        )];

        this.nameSuggestions.innerHTML = '';
        
        uniqueNames.forEach(name => {
            const button = document.createElement('button');
            button.className = 'px-3 py-1 bg-gray-100 text-gray-700 rounded-md text-sm hover:bg-gray-200 border border-gray-300 transition-colors';
            button.textContent = name;
            button.addEventListener('click', () => {
                this.personNameInput.value = name;
            });
            this.nameSuggestions.appendChild(button);
        });
    }

    async confirmFace() {
        const personName = this.personNameInput.value.trim();
        
        if (!personName) {
            alert('Please enter a person name');
            return;
        }

        if (!this.currentFaceData) {
            return;
        }

        try {
            let endpoint;
            let payload;
            
            // Check if this is an unrecognized face from Smart Recognition mode
            if (this.currentFaceData.recognitionStatus === 'unrecognized' && this.currentFaceData.faceEncoding) {
                // Use save-face endpoint for Smart Recognition unrecognized faces
                endpoint = '/face-recognition/save-face';
                payload = {
                    user_id: this.currentAnalysisData.user_id,
                    name: personName,
                    encoding: this.currentFaceData.faceEncoding
                };
            } else {
                // Use save-person-label endpoint for regular person detections
                endpoint = '/face-recognition/save-person-label';
                payload = {
                    user_id: this.currentAnalysisData.user_id,
                    video_path: this.currentFaceData.videoPath || this.currentAnalysisData.video_path,
                    frame_number: this.currentFaceData.frameNumber,
                    bbox: this.currentFaceData.bbox,
                    person_name: personName,
                    person_id: this.currentFaceData.personId || 0,
                    detection_type: this.currentFaceData.detectionType || 'person'
                };
            }
            
            // Call the appropriate API endpoint
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Failed to save face: ${response.statusText}`);
            }

            // Update local data
            this.currentFaceData.labeled = true;
            this.currentFaceData.personName = personName;
            this.currentFaceData.status = 'labeled';

            // Update UI
            this.updateLabeledCount();
            this.renderFaceGrid();
            this.closeModal();

            console.log(`Face confirmed for ${personName}`);

        } catch (error) {
            console.error('Failed to confirm face:', error);
            alert(`Failed to save face: ${error.message}`);
        }
    }

    skipFace() {
        if (!this.currentFaceData) {
            return;
        }

        this.currentFaceData.status = 'skipped';
        this.renderFaceGrid();
        this.closeModal();
    }

    rejectFace() {
        if (!this.currentFaceData) {
            return;
        }

        this.currentFaceData.status = 'rejected';
        this.renderFaceGrid();
        this.closeModal();
    }

    async saveAllLabels() {
        const unlabeledFaces = this.faceData.filter(face => face.status === 'unlabeled');
        
        if (unlabeledFaces.length === 0) {
            alert('No unlabeled faces to process');
            return;
        }

        const proceed = confirm(`This will save all ${unlabeledFaces.length} unlabeled faces as "Unknown". Continue?`);
        if (!proceed) {
            return;
        }

        this.saveAllBtn.disabled = true;
        this.saveAllBtn.textContent = 'Saving...';

        let saved = 0;
        let failed = 0;

        for (const face of unlabeledFaces) {
            try {
                let endpoint;
                let payload;
                
                // Check if this is an unrecognized face from Smart Recognition mode
                if (face.recognitionStatus === 'unrecognized' && face.faceEncoding) {
                    // Use save-face endpoint for Smart Recognition unrecognized faces
                    endpoint = '/face-recognition/save-face';
                    payload = {
                        user_id: this.currentAnalysisData.user_id,
                        name: 'Unknown',
                        encoding: face.faceEncoding
                    };
                } else {
                    // Use save-person-label endpoint for regular person detections
                    endpoint = '/face-recognition/save-person-label';
                    payload = {
                        user_id: this.currentAnalysisData.user_id,
                        video_path: face.videoPath || this.currentAnalysisData.video_path,
                        frame_number: face.frameNumber,
                        bbox: face.bbox,
                        person_name: 'Unknown',
                        person_id: face.personId || 0,
                        detection_type: face.detectionType || 'person'
                    };
                }
                
                // Call the API endpoint
                const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`Failed to save face: ${response.statusText}`);
                }
                
                face.labeled = true;
                face.personName = 'Unknown';
                face.status = 'labeled';
                saved++;
            } catch (error) {
                console.error('Failed to save face:', error);
                failed++;
            }
        }

        this.saveAllBtn.disabled = false;
        this.saveAllBtn.textContent = 'Save All Labels';

        this.updateLabeledCount();
        this.renderFaceGrid();

        alert(`Saved ${saved} faces. ${failed} failed.`);
    }

    async confirmBulkLabels() {
        // Handle both old bulk labeling and new consolidated labeling
        const isConsolidatedMode = this.currentFrameGroup !== undefined;
        
        if (!this.currentFrameDetections && !this.currentFrameGroup) {
            console.error('No frame detections or frame group available for bulk labeling');
            return;
        }

        // Get all person inputs - handles both old and new input structures
        const personInputs = this.bulkPersonInputs.querySelectorAll('input[type="text"]');
        const personLabels = [];

        // Collect labels from inputs
        personInputs.forEach(input => {
            const personName = input.value.trim();
            if (personName) {
                let personId, bbox, detectionType;
                
                if (isConsolidatedMode) {
                    // New consolidated interface data attributes
                    personId = parseInt(input.dataset.personId) || 0;
                    detectionType = input.dataset.detectionType || 'person';
                    bbox = {
                        x1: parseInt(input.dataset.bboxX1),
                        y1: parseInt(input.dataset.bboxY1),
                        x2: parseInt(input.dataset.bboxX2),
                        y2: parseInt(input.dataset.bboxY2)
                    };
                } else {
                    // Old bulk interface data attributes
                    personId = parseInt(input.dataset.personId);
                    bbox = JSON.parse(input.dataset.bbox);
                    detectionType = input.dataset.detectionType;
                }

                personLabels.push({
                    person_id: personId,
                    person_name: personName,
                    bbox: bbox,
                    detection_type: detectionType
                });
            }
        });

        if (personLabels.length === 0) {
            alert('Please enter at least one person name');
            return;
        }

        try {
            // Determine video path and frame number
            const videoPath = isConsolidatedMode ? 
                this.currentFrameGroup.videoPath : 
                (this.currentFaceData.videoPath || this.currentAnalysisData.video_path);
            const frameNumber = isConsolidatedMode ? 
                this.currentFrameGroup.frameNumber : 
                this.currentFaceData.frameNumber;

            // Call bulk labeling API
            const payload = {
                user_id: this.currentAnalysisData.user_id,
                video_path: videoPath,
                frame_number: frameNumber,
                person_labels: personLabels
            };

            console.log('Sending bulk labeling payload:', payload);

            const response = await fetch(`${this.apiBaseUrl}/face-recognition/save-bulk-person-labels`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Failed to save bulk labels: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('Bulk labeling result:', result);

            // Update local data for all labeled persons
            personLabels.forEach(personLabel => {
                if (isConsolidatedMode) {
                    // Update detections in frame group
                    this.currentFrameGroup.detections.forEach(detection => {
                        if (detection.personId === personLabel.person_id && 
                            detection.detectionType === personLabel.detection_type) {
                            detection.labeled = true;
                            detection.personName = personLabel.person_name;
                            detection.status = 'labeled';
                        }
                    });
                    
                    // Also update faceData for consistency
                    this.faceData.forEach(face => {
                        if (face.frameNumber === this.currentFrameGroup.frameNumber &&
                            face.personId === personLabel.person_id &&
                            face.detectionType === personLabel.detection_type) {
                            face.labeled = true;
                            face.personName = personLabel.person_name;
                            face.status = 'labeled';
                        }
                    });
                } else {
                    // Old bulk interface - find and update face data matching this frame
                    const matchingFaces = this.faceData.filter(face => 
                        face.frameNumber === this.currentFaceData.frameNumber &&
                        face.videoPath === (this.currentFaceData.videoPath || this.currentAnalysisData.video_path)
                    );

                    matchingFaces.forEach(face => {
                        face.labeled = true;
                        face.personName = personLabel.person_name;
                        face.status = 'labeled';
                    });
                }
            });

            // Update UI
            this.updateLabeledCount();
            this.renderFaceGrid();
            this.closeModal();

            // Show success message
            alert(`Successfully saved labels for ${personLabels.length} detection${personLabels.length > 1 ? 's' : ''}!`);

        } catch (error) {
            console.error('Failed to save bulk labels:', error);
            alert(`Failed to save bulk labels: ${error.message}`);
        }
    }

    skipBulkLabels() {
        if (!this.currentFaceData) {
            return;
        }

        // Mark face as skipped
        this.currentFaceData.status = 'skipped';
        this.renderFaceGrid();
        this.closeModal();
    }

    setAnalyzing(analyzing) {
        this.analyzeBtn.disabled = analyzing;
        if (analyzing) {
            this.analyzeText.textContent = 'Analyzing...';
            this.analyzeSpinner.classList.remove('hidden');
        } else {
            this.analyzeText.textContent = 'Start Analysis';
            this.analyzeSpinner.classList.add('hidden');
        }
    }

    showLoading() {
        this.loadingState.classList.remove('hidden');
    }

    hideLoading() {
        this.loadingState.classList.add('hidden');
    }

    showResults() {
        this.resultsSection.classList.remove('hidden');
    }

    hideResults() {
        this.resultsSection.classList.add('hidden');
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorState.classList.remove('hidden');
    }

    hideError() {
        this.errorState.classList.add('hidden');
    }

    navigateToSegmentation() {
        if (!this.currentAnalysisData) {
            alert('Please complete face recognition analysis first');
            return;
        }

        const labeledCount = this.faceData.filter(face => face.labeled).length;
        if (labeledCount === 0) {
            alert('Please label at least one face before continuing to segmentation');
            return;
        }

        // Navigate to segmentation interface with current data
        const params = new URLSearchParams({
            video_path: this.currentAnalysisData.video_path || this.videoPathInput.value.trim(),
            user_id: this.currentAnalysisData.user_id,
            service_url: this.apiBaseUrl,
            faces_count: labeledCount.toString()
        });

        window.location.href = `../segmentation/index.html?${params.toString()}`;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FaceRecognitionUI();
});