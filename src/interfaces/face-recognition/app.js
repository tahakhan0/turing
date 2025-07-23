// Turing Face Recognition Web UI - Main Application JavaScript

class FaceRecognitionUI {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentAnalysisData = null;
        this.faceData = [];
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
        let faceId = 0;

        console.log('Processing face data from:', analysisData);
        
        // Check if this is a recognition analysis (has detections with frame-by-frame data)
        if (analysisData.detections !== undefined && analysisData.total_frames !== undefined) {
            // New face recognition analysis format
            console.log('Processing face recognition analysis with detections:', analysisData);
            
            analysisData.detections.forEach(frameData => {
                frameData.detections.forEach(detection => {
                    const isRecognized = detection.recognition_status === 'recognized';
                    
                    this.faceData.push({
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
                        faceEncoding: detection.face_encoding // Store for potential saving
                    });
                });
            });
        } else if (analysisData.unique_persons !== undefined) {
            // Directory analysis format
            console.log('Processing directory analysis with unique persons:', analysisData.unique_persons);
            
            analysisData.detections.forEach(detection => {
                console.log('Processing unique person detection:', detection);
                
                this.faceData.push({
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
                    status: 'unlabeled'
                });
            });
        } else if (analysisData.analysis !== undefined) {
            // Single video analysis format (VideoAnalysis)
            console.log('Processing single video analysis, analysis array:', analysisData.analysis);
            
            analysisData.analysis.forEach(frameAnalysis => {
                console.log('Processing frame analysis:', frameAnalysis);
                // Include both face and person detections for now
                const faceDetections = frameAnalysis.detections.filter(d => d.class_name === 'person');
                console.log('Person detections in this frame:', faceDetections);
                
                faceDetections.forEach(detection => {
                    this.faceData.push({
                        id: faceId++,
                        frameNumber: frameAnalysis.frame_number,
                        timestamp: null,
                        bbox: detection.bbox,
                        confidence: detection.confidence,
                        visualizationUrl: frameAnalysis.visualization_url,
                        videoName: analysisData.video_path || 'Unknown',
                        videoPath: analysisData.video_path,
                        labeled: false,
                        personName: '',
                        status: 'unlabeled' // unlabeled, labeled, skipped, rejected
                    });
                });
            });
        }

        console.log(`Processed ${this.faceData.length} faces from analysis`);
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
    }

    renderFaceGrid() {
        const filteredFaces = this.getFilteredFaces();
        
        console.log('Rendering face grid with', filteredFaces.length, 'faces');
        console.log('Filtered faces:', filteredFaces);
        
        this.facesGrid.innerHTML = '';

        if (filteredFaces.length === 0) {
            console.log('No faces to display');
            this.facesGrid.innerHTML = `
                <div class="col-span-full text-center py-8 text-gray-500">
                    No faces match the current filter
                </div>
            `;
            return;
        }

        filteredFaces.forEach(face => {
            console.log('Creating card for face:', face);
            const faceCard = this.createFaceCard(face);
            this.facesGrid.appendChild(faceCard);
        });
    }

    createFaceCard(face) {
        const card = document.createElement('div');
        card.className = `face-card bg-white border border-gray-200 rounded-lg overflow-hidden cursor-pointer hover:shadow-md transition-all ${face.labeled ? 'border-green-300' : ''}`;
        
        const statusColor = {
            'unlabeled': 'bg-gray-100 text-gray-700',
            'labeled': 'bg-green-100 text-green-700',
            'skipped': 'bg-yellow-100 text-yellow-700',
            'rejected': 'bg-red-100 text-red-700'
        }[face.status];

        card.innerHTML = `
            <div class="aspect-square bg-gray-50 flex items-center justify-center">
                ${face.visualizationUrl ? 
                    `<img src="${face.visualizationUrl}" alt="Face detection" class="w-full h-full object-cover">` :
                    `<div class="text-gray-400">No Image</div>`
                }
            </div>
            <div class="p-4">
                <div class="flex justify-between items-start mb-2">
                    <div class="text-sm text-gray-600">Frame ${face.frameNumber}</div>
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusColor}">
                        ${face.status.charAt(0).toUpperCase() + face.status.slice(1)}
                    </span>
                </div>
                ${face.videoName ? `<div class="text-xs text-gray-400 mb-1">${face.videoName}</div>` : ''}
                <div class="text-sm text-gray-500 mb-2">
                    Confidence: ${(face.confidence * 100).toFixed(1)}%
                </div>
                ${face.personName ? 
                    `<div class="text-sm font-medium text-gray-900">${face.personName}</div>` :
                    `<div class="text-sm text-gray-400">Click to label</div>`
                }
            </div>
        `;

        card.addEventListener('click', () => this.openLabelingModal(face));
        return card;
    }

    getFilteredFaces() {
        let filtered = [...this.faceData];

        // Filter by status
        const statusFilter = this.filterStatus.value;
        if (statusFilter !== 'all') {
            filtered = filtered.filter(face => face.status === statusFilter);
        }

        // Filter by name search
        const searchTerm = this.searchNames.value.toLowerCase().trim();
        if (searchTerm) {
            filtered = filtered.filter(face => 
                face.personName.toLowerCase().includes(searchTerm)
            );
        }

        return filtered;
    }

    filterFaces() {
        this.renderFaceGrid();
    }

    async openLabelingModal(face) {
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
        
        // Set frame number
        this.modalFrameNumber.textContent = face.frameNumber;
        
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
                    person_name: personName
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
                        person_name: 'Unknown'
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
        if (!this.currentFrameDetections) {
            console.error('No frame detections available for bulk labeling');
            return;
        }

        // Get all person inputs
        const personInputs = this.bulkPersonInputs.querySelectorAll('input[type="text"]');
        const personLabels = [];

        // Collect labels from inputs
        personInputs.forEach(input => {
            const personName = input.value.trim();
            if (personName) {
                const personId = parseInt(input.dataset.personId);
                const bbox = JSON.parse(input.dataset.bbox);
                const detectionType = input.dataset.detectionType;

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
            // Call bulk labeling API
            const payload = {
                user_id: this.currentAnalysisData.user_id,
                video_path: this.currentFaceData.videoPath || this.currentAnalysisData.video_path,
                frame_number: this.currentFaceData.frameNumber,
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
                // Find and update face data matching this frame
                const matchingFaces = this.faceData.filter(face => 
                    face.frameNumber === this.currentFaceData.frameNumber &&
                    face.videoPath === (this.currentFaceData.videoPath || this.currentAnalysisData.video_path)
                );

                matchingFaces.forEach(face => {
                    face.labeled = true;
                    face.personName = personLabel.person_name;
                    face.status = 'labeled';
                });
            });

            // Update UI
            this.updateLabeledCount();
            this.renderFaceGrid();
            this.closeModal();

            // Show success message
            alert(`Successfully saved labels for ${personLabels.length} people!`);

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
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FaceRecognitionUI();
});