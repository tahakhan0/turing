class FaceRecognitionUI {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.userId = localStorage.getItem('alan_user_id');
        this.folderPath = '/Users/tahakhan/Desktop/ring_camera/my_demo_folder';
        this.faceData = [];
        this.frameGroups = new Map();
        this.analysisData = null;

        this.initializeElements();

        if (!this.userId) {
            alert('User session not found. Redirecting to onboarding.');
            window.location.href = '../onboarding/index.html';
            return;
        }

        this.checkServiceConnection();
    }

    initializeElements() {
        // Navigation elements
        this.backToOnboardingBtn = document.getElementById('back-to-onboarding');
        this.continueToSegmentationBtn = document.getElementById('continue-to-segmentation');
        this.connectionStatus = document.getElementById('connection-status');

        // State elements
        this.downloadingState = document.getElementById('downloading-state');
        this.progressText = document.getElementById('progress-text');

        this.analyzingState = document.getElementById('analyzing-state');
        this.analysisPercentage = document.getElementById('analysis-percentage');

        this.labelingSection = document.getElementById('labeling-section');
        this.facesGrid = document.getElementById('faces-grid');
        this.continueBtn = document.getElementById('continue-btn');

        // Modal elements
        this.modal = document.getElementById('labeling-modal');
        this.modalTitle = document.getElementById('modal-title');
        this.modalFaceImage = document.getElementById('modal-face-image');
        this.modalFrameNumber = document.getElementById('modal-frame-number');
        this.closeModalBtn = document.getElementById('close-modal');
        this.bulkLabelingForm = document.getElementById('bulk-labeling-form');
        this.bulkPersonInputs = document.getElementById('bulk-person-inputs');
        this.confirmBulkBtn = document.getElementById('confirm-bulk');
        this.skipBulkBtn = document.getElementById('skip-bulk');

        // Event listeners
        this.backToOnboardingBtn.addEventListener('click', () => this.navigateToOnboarding());
        this.continueToSegmentationBtn.addEventListener('click', () => this.navigateToNextStep());
        this.continueBtn.addEventListener('click', () => this.navigateToNextStep());
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        this.confirmBulkBtn.addEventListener('click', () => this.confirmBulkLabels());
        this.skipBulkBtn.addEventListener('click', () => this.closeModal());
    }

    async checkServiceConnection() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (!response.ok) throw new Error('Service unavailable');
            
            // Update connection status
            const statusDot = this.connectionStatus.querySelector('.w-2');
            const statusText = this.connectionStatus.querySelector('span');
            statusDot.classList.remove('bg-red-400');
            statusDot.classList.add('bg-green-400');
            statusText.textContent = 'Connected';
            
            this.startProcess();
        } catch (error) {
            console.error('Service connection failed:', error);
            
            // Update connection status
            const statusDot = this.connectionStatus.querySelector('.w-2');
            const statusText = this.connectionStatus.querySelector('span');
            statusDot.classList.remove('bg-green-400');
            statusDot.classList.add('bg-red-400');
            statusText.textContent = 'Disconnected';
            
            this.downloadingState.innerHTML = '<div class="text-center"><div class="inline-flex items-center justify-center p-3 bg-red-500 rounded-md shadow-lg mb-4"><svg class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg></div><p class="text-red-500 text-lg font-medium">Unable to connect to analysis service</p></div>';
        }
    }

    startProcess() {
        this.simulateDownload();
    }

    simulateDownload() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 100) progress = 100;
            this.progressText.textContent = `${Math.round(progress)}%`;

            if (progress === 100) {
                clearInterval(interval);
                setTimeout(() => {
                    this.downloadingState.classList.add('hidden');
                    this.analyzingState.classList.remove('hidden');
                    this.startAnalysis();
                }, 500);
            }
        }, 300);
    }

    async startAnalysis() {
        const payload = { user_id: this.userId, folder_path: this.folderPath };

        try {
            // Use enrollment endpoint for folder processing (processes all videos)
            this.analysisPercentage.textContent = '50%';
            const enrollmentResponse = await fetch(`${this.apiBaseUrl}/face-recognition/enrollment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!enrollmentResponse.ok) throw new Error(`Enrollment failed! status: ${enrollmentResponse.status}`);

            this.analysisPercentage.textContent = '100%';
            const data = await enrollmentResponse.json();
            this.analysisData = data;
            this.processFaceData(data);
            this.displayLabelingUI();

        } catch (error) {
            console.error('Analysis failed:', error);
            this.analyzingState.innerHTML = `<p class="text-red-500">Analysis Failed: ${error.message}</p>`;
        }
    }

    processFaceData(analysisData) {
        this.faceData = [];
        this.frameGroups = new Map();
        let faceId = 0;

        // Handle FaceRecognitionAnalysis format from /analyze endpoint
        if (analysisData.detections && Array.isArray(analysisData.detections)) {
            // This is FaceRecognitionAnalysis format
            analysisData.detections.forEach(frameData => {
                if (frameData.detections && frameData.detections.length > 0) {
                    const frameKey = `${analysisData.video_path}_${frameData.frame_number}`;
                    if (!this.frameGroups.has(frameKey)) {
                        this.frameGroups.set(frameKey, {
                            frameNumber: frameData.frame_number,
                            detections: [],
                            visualizationUrl: frameData.visualization_url,
                            videoPath: analysisData.video_path,
                            videoName: analysisData.video_path ? analysisData.video_path.split('/').pop() : 'Unknown'
                        });
                    }

                    frameData.detections.forEach((detection, index) => {
                        const isLabeled = detection.person_name && detection.person_name.trim() !== '';
                        const faceData = {
                            id: faceId++,
                            frameNumber: frameData.frame_number,
                            bbox: detection.bbox,
                            confidence: detection.confidence || 0.0,
                            visualizationUrl: frameData.visualization_url,
                            videoPath: analysisData.video_path,
                            labeled: isLabeled,
                            personName: detection.person_name || '',
                            status: isLabeled ? 'labeled' : 'unlabeled',
                            detectionType: detection.detection_type || 'person',
                            personId: detection.person_id !== undefined ? detection.person_id : index,
                            recognized: detection.recognition_status === 'recognized',
                            confidence_score: detection.confidence || 0.0
                        };
                        this.faceData.push(faceData);
                        this.frameGroups.get(frameKey).detections.push(faceData);
                    });
                }
            });
        }
        // Handle legacy VideoAnalysis format from /enrollment endpoint
        else {
            const analysisResults = analysisData.analysis || analysisData.frame_results || [];
            
            analysisResults.forEach(frameAnalysis => {
                if (frameAnalysis.detections && frameAnalysis.detections.length > 0) {
                    const frameKey = `${frameAnalysis.video_path}_${frameAnalysis.frame_number}`;
                    if (!this.frameGroups.has(frameKey)) {
                        this.frameGroups.set(frameKey, {
                            frameNumber: frameAnalysis.frame_number,
                            detections: [],
                            visualizationUrl: frameAnalysis.visualization_url,
                            videoPath: frameAnalysis.video_path,
                            videoName: frameAnalysis.video_path ? frameAnalysis.video_path.split('/').pop() : 'Unknown'
                        });
                    }

                    frameAnalysis.detections.forEach((detection, index) => {
                        const isLabeled = detection.person_name && detection.person_name.trim() !== '';
                        const faceData = {
                            id: faceId++,
                            frameNumber: frameAnalysis.frame_number,
                            bbox: detection.bbox,
                            confidence: detection.confidence,
                            visualizationUrl: frameAnalysis.visualization_url,
                            videoPath: frameAnalysis.video_path,
                            labeled: isLabeled,
                            personName: detection.person_name || '',
                            status: isLabeled ? 'labeled' : 'unlabeled',
                            detectionType: detection.detection_type || 'person',
                            personId: detection.person_id !== undefined ? detection.person_id : index,
                            recognized: detection.recognized || false,
                            confidence_score: detection.confidence_score || detection.confidence
                        };
                        this.faceData.push(faceData);
                        this.frameGroups.get(frameKey).detections.push(faceData);
                    });
                }
            });
        }
    }

    displayLabelingUI() {
        this.analyzingState.classList.add('hidden');
        this.labelingSection.classList.remove('hidden');
        this.renderFaceGrid();
    }

    renderFaceGrid() {
        this.facesGrid.innerHTML = '';
        this.frameGroups.forEach(frameGroup => {
            const faceCard = this.createFaceCard(frameGroup);
            this.facesGrid.appendChild(faceCard);
        });
    }

    createFaceCard(frameGroup) {
        const card = document.createElement('div');
        const allLabeled = frameGroup.detections.every(d => d.status === 'labeled');
        const anyLabeled = frameGroup.detections.some(d => d.status === 'labeled');
        const statusText = allLabeled ? 'Labeled' : anyLabeled ? 'Partially Labeled' : 'Unlabeled';
        
        let statusColorClass = 'bg-red-100 text-red-800';
        let statusBorderClass = 'border-gray-300';
        if (allLabeled) {
            statusColorClass = 'bg-green-100 text-green-800';
            statusBorderClass = 'border-green-400';
        } else if (anyLabeled) {
            statusColorClass = 'bg-yellow-100 text-yellow-800';
            statusBorderClass = 'border-yellow-400';
        }

        card.className = `face-card bg-white border ${statusBorderClass} rounded-lg overflow-hidden cursor-pointer hover:shadow-lg transition-all transform hover:-translate-y-1`;

        card.innerHTML = `
            <div class="aspect-w-16 aspect-h-10 bg-gray-200">
                <img src="${frameGroup.visualizationUrl}" alt="Frame ${frameGroup.frameNumber}" class="w-full h-full object-cover">
            </div>
            <div class="p-5">
                <p class="text-base font-semibold text-gray-900">Frame ${frameGroup.frameNumber}</p>
                <p class="text-sm text-gray-600 mb-3">${frameGroup.detections.length} detections</p>
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusColorClass}">
                    ${statusText}
                </span>
            </div>
        `;
        card.addEventListener('click', () => this.openLabelingModal(frameGroup));
        return card;
    }

    openLabelingModal(frameGroup) {
        this.currentFrameGroup = frameGroup;
        this.modalTitle.textContent = `Label Detections in Frame ${frameGroup.frameNumber}`;
        this.modalFaceImage.src = frameGroup.visualizationUrl;
        this.modalFrameNumber.textContent = frameGroup.frameNumber;
        this.createConsolidatedPersonInputs(frameGroup);
        this.modal.classList.remove('hidden');
    }

    createConsolidatedPersonInputs(frameGroup) {
        this.bulkPersonInputs.innerHTML = '';
        const colorNames = [
            'Green', 'Red', 'Blue', 'Cyan', 'Magenta', 'Yellow', 
            'Purple', 'Orange', 'Light Blue', 'Lime', 'Deep Pink', 'Dark Turquoise'
        ];

        frameGroup.detections.forEach((detection, index) => {
            const colorName = colorNames[detection.personId % colorNames.length];
            const inputDiv = document.createElement('div');
            inputDiv.className = 'mb-4';
            inputDiv.innerHTML = `
                <label class="block text-sm font-medium text-gray-700 mb-1">Label Person (${colorName} Box)</label>
                <input type="text" value="${detection.personName || ''}" data-detection-index="${index}" 
                       class="consolidated-person-input w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-brand-blue focus:border-brand-blue">
            `;
            this.bulkPersonInputs.appendChild(inputDiv);
        });
    }

    async confirmBulkLabels() {
        const personInputs = this.bulkPersonInputs.querySelectorAll('.consolidated-person-input');
        const personLabels = [];

        personInputs.forEach(input => {
            const personName = input.value.trim();
            if (personName) {
                const detectionIndex = parseInt(input.dataset.detectionIndex);
                const detection = this.currentFrameGroup.detections[detectionIndex];
                personLabels.push({
                    person_id: detection.personId,
                    person_name: personName,
                    bbox: detection.bbox,
                    detection_type: detection.detectionType
                });
            }
        });

        if (personLabels.length === 0) {
            this.closeModal();
            return;
        }

        const payload = {
            user_id: this.userId,
            video_path: this.currentFrameGroup.videoPath,
            frame_number: this.currentFrameGroup.frameNumber,
            person_labels: personLabels
        };

        try {
            const response = await fetch(`${this.apiBaseUrl}/face-recognition/save-bulk-person-labels`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Bulk labels saved:', result);

            // Update local data
            personLabels.forEach(label => {
                const detection = this.currentFrameGroup.detections.find(d => d.personId === label.person_id);
                if (detection) {
                    detection.personName = label.person_name;
                    detection.status = 'labeled';
                    detection.labeled = true;
                }
            });

            this.renderFaceGrid();
            this.updateContinueButton();
            this.closeModal();
            this.showToast('Labels saved successfully!');

        } catch (error) {
            console.error('Failed to save labels:', error);
            this.showToast('Failed to save labels. Please try again.', 'error');
        }
    }

    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        const bgColor = type === 'success' ? 'bg-green-600' : 'bg-red-600';
        toast.className = `fixed bottom-5 right-5 text-white px-6 py-3 rounded-lg shadow-lg transform translate-y-20 opacity-0 transition-all duration-300 ease-out ${bgColor}`;
        toast.textContent = message;

        document.body.appendChild(toast);

        // Animate in
        setTimeout(() => {
            toast.classList.remove('translate-y-20');
            toast.classList.remove('opacity-0');
        }, 10);

        // Animate out and remove
        setTimeout(() => {
            toast.classList.add('opacity-0');
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, 4000);
    }

    updateContinueButton() {
        const anyLabeled = this.faceData.some(face => face.status === 'labeled' || face.labeled);
        this.continueBtn.disabled = !anyLabeled;
        this.continueToSegmentationBtn.disabled = !anyLabeled;
        
        if (anyLabeled) {
            this.continueBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            this.continueBtn.classList.add('hover:bg-brand-purple');
            this.continueToSegmentationBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            this.continueToSegmentationBtn.classList.add('hover:bg-brand-purple');
        } else {
            this.continueBtn.classList.add('opacity-50', 'cursor-not-allowed');
            this.continueBtn.classList.remove('hover:bg-brand-purple');
            this.continueToSegmentationBtn.classList.add('opacity-50', 'cursor-not-allowed');
            this.continueToSegmentationBtn.classList.remove('hover:bg-brand-purple');
        }
    }

    closeModal() {
        this.modal.classList.add('hidden');
        this.currentFrameGroup = null;
    }

    navigateToOnboarding() {
        window.location.href = '../onboarding/index.html';
    }

    navigateToNextStep() {
        const params = new URLSearchParams({
            user_id: this.userId,
            service_url: this.apiBaseUrl,
            auto_start: 'true'
        });
        window.location.href = `../segmentation/index.html?${params.toString()}`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new FaceRecognitionUI();
});