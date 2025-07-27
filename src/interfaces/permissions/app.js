// Alan AI Access Permissions Web UI - Main Application JavaScript

class AccessPermissionsUI {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentVideoPath = null;
        this.currentUserId = null;
        this.labeledPeople = [];
        this.verifiedAreas = [];
        this.permissions = new Map(); // Map of person -> area -> permission
        this.selectedPerson = null;
        this.currentModalArea = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadDataFromUrl();
        this.loadInitialData();
    }

    initializeElements() {
        // Navigation elements
        this.backToSegmentationBtn = document.getElementById('back-to-segmentation');
        this.finishSetupBtn = document.getElementById('finish-setup');

        // Summary elements
        this.totalPeopleElement = document.getElementById('total-people');
        this.totalAreasElement = document.getElementById('total-areas');
        this.totalPermissionsElement = document.getElementById('total-permissions');
        this.completionRateElement = document.getElementById('completion-rate');

        // Main interface elements
        this.loadingState = document.getElementById('loading-state');
        this.permissionsInterface = document.getElementById('permissions-interface');
        this.errorState = document.getElementById('error-state');
        this.errorMessage = document.getElementById('error-message');

        // People selection
        this.peopleGrid = document.getElementById('people-grid');

        // Permissions configuration
        this.permissionsConfig = document.getElementById('permissions-config');
        this.selectedPersonName = document.getElementById('selected-person-name');
        this.grantAllAccessBtn = document.getElementById('grant-all-access');
        this.revokeAllAccessBtn = document.getElementById('revoke-all-access');
        this.globalAccessToggle = document.getElementById('global-access-toggle');
        this.areaPermissionsList = document.getElementById('area-permissions-list');

        // Quick action buttons
        this.presetAdultsBtn = document.getElementById('preset-adults');
        this.presetChildrenBtn = document.getElementById('preset-children');
        this.presetGuestsBtn = document.getElementById('preset-guests');

        // Modal elements
        this.modal = document.getElementById('permission-modal');
        this.modalTitle = document.getElementById('modal-title');
        this.modalAreaType = document.getElementById('modal-area-type');
        this.modalAreaSize = document.getElementById('modal-area-size');
        this.modalPersonName = document.getElementById('modal-person-name');
        this.modalCurrentAccess = document.getElementById('modal-current-access');
        this.closeModalBtn = document.getElementById('close-modal');
        this.savePermissionBtn = document.getElementById('save-permission');
        this.cancelPermissionBtn = document.getElementById('cancel-permission');
        this.conditionsSection = document.getElementById('conditions-section');
    }

    attachEventListeners() {
        // Navigation
        this.backToSegmentationBtn.addEventListener('click', () => this.navigateToSegmentation());
        this.finishSetupBtn.addEventListener('click', () => this.finishSetup());

        // Global access toggle
        this.globalAccessToggle.addEventListener('change', (e) => this.handleGlobalAccessToggle(e));

        // Quick actions
        this.grantAllAccessBtn.addEventListener('click', () => this.grantAllAccess());
        this.revokeAllAccessBtn.addEventListener('click', () => this.revokeAllAccess());

        // Preset buttons
        this.presetAdultsBtn.addEventListener('click', () => this.applyAdultPreset());
        this.presetChildrenBtn.addEventListener('click', () => this.applyChildPreset());
        this.presetGuestsBtn.addEventListener('click', () => this.applyGuestPreset());

        // Modal
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        this.cancelPermissionBtn.addEventListener('click', () => this.closeModal());
        this.savePermissionBtn.addEventListener('click', () => this.savePermission());

        // Access level radio buttons
        document.addEventListener('change', (e) => {
            if (e.target.name === 'access-level') {
                this.conditionsSection.style.display = e.target.value === 'allowed' ? 'block' : 'none';
            }
        });

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
        const serviceUrl = urlParams.get('service_url');
        
        if (serviceUrl) {
            this.apiBaseUrl = serviceUrl;
        }
    }

    async loadInitialData() {
        try {
            await this.checkServiceConnection();
            await this.loadLabeledPeople();
            await this.loadVerifiedAreas();
            await this.loadExistingPermissions();
            
            this.displayInterface();
            this.updateSummary();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError(error.message);
        } finally {
            this.loadingState.classList.add('hidden');
        }
    }

    async checkServiceConnection() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/segmentation/health`);
            this.updateConnectionStatus(response.ok);
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

    async loadLabeledPeople() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/segmentation/people/${this.currentUserId}`);
            if (!response.ok) {
                throw new Error(`Failed to load labeled people: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.labeledPeople = data.people || [];
            
        } catch (error) {
            console.error('Error loading labeled people:', error);
            throw new Error('Could not load labeled people from face recognition data');
        }
    }

    async loadVerifiedAreas() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/segmentation/segments/user/${this.currentUserId}`);
            if (!response.ok) {
                throw new Error(`Failed to load areas: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.verifiedAreas = (data.segments || []).filter(area => area.verified);
            
        } catch (error) {
            console.error('Error loading verified areas:', error);
            throw new Error('Could not load verified areas from segmentation data');
        }
    }

    async loadExistingPermissions() {
        this.permissions.clear();
        
        for (const person of this.labeledPeople) {
            try {
                const response = await fetch(`${this.apiBaseUrl}/segmentation/permissions/person/${encodeURIComponent(person)}/user/${this.currentUserId}`);
                if (response.ok) {
                    const data = await response.json();
                    const personPermissions = new Map();
                    
                    for (const perm of data.permissions || []) {
                        personPermissions.set(perm.area_id, {
                            allowed: perm.allowed,
                            conditions: perm.conditions || []
                        });
                    }
                    
                    this.permissions.set(person, personPermissions);
                }
            } catch (error) {
                console.error(`Error loading permissions for ${person}:`, error);
            }
        }
    }

    displayInterface() {
        this.permissionsInterface.classList.remove('hidden');
        this.displayPeopleGrid();
    }

    displayPeopleGrid() {
        this.peopleGrid.innerHTML = '';

        this.labeledPeople.forEach(person => {
            const personCard = this.createPersonCard(person);
            this.peopleGrid.appendChild(personCard);
        });
    }

    createPersonCard(person) {
        const card = document.createElement('div');
        card.className = `person-card p-4 border-2 border-gray-200 rounded-lg cursor-pointer hover:border-brand-blue hover:bg-gray-50 transition-all`;
        
        const personPermissions = this.permissions.get(person) || new Map();
        const totalPermissions = personPermissions.size;
        const allowedPermissions = Array.from(personPermissions.values()).filter(p => p.allowed).length;
        
        card.innerHTML = `
            <div class="text-center">
                <div class="w-16 h-16 bg-brand-purple rounded-full flex items-center justify-center mx-auto mb-3">
                    <span class="text-white text-2xl font-bold">${person.charAt(0).toUpperCase()}</span>
                </div>
                <h3 class="font-medium text-gray-900 mb-1">${person}</h3>
                <p class="text-sm text-gray-600">${allowedPermissions}/${this.verifiedAreas.length} areas</p>
                <div class="mt-2">
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div class="bg-brand-blue h-2 rounded-full" style="width: ${this.verifiedAreas.length > 0 ? (allowedPermissions / this.verifiedAreas.length) * 100 : 0}%"></div>
                    </div>
                </div>
            </div>
        `;

        card.addEventListener('click', () => this.selectPerson(person, card));
        return card;
    }

    selectPerson(person, cardElement) {
        // Update UI selection
        document.querySelectorAll('.person-card').forEach(card => {
            card.classList.remove('selected');
        });
        cardElement.classList.add('selected');

        this.selectedPerson = person;
        this.selectedPersonName.textContent = person;
        this.permissionsConfig.classList.remove('hidden');

        this.displayPersonPermissions();
    }

    displayPersonPermissions() {
        const personPermissions = this.permissions.get(this.selectedPerson) || new Map();
        
        // Update global access toggle
        const hasGlobalAccess = personPermissions.has('all') && personPermissions.get('all').allowed;
        this.globalAccessToggle.checked = hasGlobalAccess;

        // Display individual area permissions
        this.areaPermissionsList.innerHTML = '';

        this.verifiedAreas.forEach(area => {
            const permissionItem = this.createAreaPermissionItem(area, personPermissions);
            this.areaPermissionsList.appendChild(permissionItem);
        });
    }

    createAreaPermissionItem(area, personPermissions) {
        const permission = personPermissions.get(area.area_id);
        const hasAccess = permission?.allowed || false;
        const conditions = permission?.conditions || [];
        const isGlobalAccess = personPermissions.has('all') && personPermissions.get('all').allowed;

        const item = document.createElement('div');
        item.className = 'area-permission-item border border-gray-200 rounded-lg p-4';
        
        item.innerHTML = `
            <div class="flex items-center justify-between">
                <div class="flex-1">
                    <div class="flex items-center space-x-3">
                        <div class="w-12 h-12 bg-gradient-to-br from-blue-100 to-purple-100 rounded-lg flex items-center justify-center">
                            <span class="text-lg">${this.getAreaIcon(area.area_type)}</span>
                        </div>
                        <div>
                            <h4 class="font-medium text-gray-900 capitalize">${area.area_type.replace('_', ' ')}</h4>
                            <p class="text-sm text-gray-600">${area.dimensions?.area?.toFixed(1) || 'N/A'} m²</p>
                            ${conditions.length > 0 ? `<p class="text-xs text-orange-600">With conditions</p>` : ''}
                            ${isGlobalAccess ? `<p class="text-xs text-blue-600">Via global access</p>` : ''}
                        </div>
                    </div>
                </div>
                <div class="flex items-center space-x-3">
                    <label class="toggle-switch">
                        <input type="checkbox" ${(hasAccess || isGlobalAccess) ? 'checked' : ''} ${isGlobalAccess ? 'disabled' : ''} 
                               data-area-id="${area.area_id}" class="area-toggle">
                        <span class="toggle-slider"></span>
                    </label>
                    <button class="config-btn bg-gray-100 hover:bg-gray-200 p-2 rounded-lg transition-colors" 
                            data-area-id="${area.area_id}">
                        <svg class="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `;

        // Add event listeners
        const toggle = item.querySelector('.area-toggle');
        const configBtn = item.querySelector('.config-btn');

        toggle.addEventListener('change', (e) => this.handleAreaToggle(e));
        configBtn.addEventListener('click', (e) => this.openPermissionModal(area));

        return item;
    }

    getAreaIcon(areaType) {
        const icons = {
            'backyard': '🌿',
            'pool': '🏊',
            'garage': '🚗',
            'road': '🛣️',
            'driveway': '🚙',
            'front_yard': '🏡',
            'lawn': '🌱',
            'patio': '🪑',
            'deck': '🏠',
            'fence': '🚧'
        };
        return icons[areaType] || '📍';
    }

    async handleAreaToggle(event) {
        const areaId = event.target.dataset.areaId;
        const allowed = event.target.checked;

        try {
            await this.setPermission(this.selectedPerson, areaId, allowed, []);
            await this.loadExistingPermissions();
            this.displayPersonPermissions();
            this.updateSummary();
        } catch (error) {
            console.error('Error updating permission:', error);
            this.showError('Failed to update permission');
            // Revert toggle
            event.target.checked = !allowed;
        }
    }

    async handleGlobalAccessToggle(event) {
        const allowed = event.target.checked;

        try {
            await this.setPermission(this.selectedPerson, 'all', allowed, []);
            await this.loadExistingPermissions();
            this.displayPersonPermissions();
            this.updateSummary();
        } catch (error) {
            console.error('Error updating global access:', error);
            this.showError('Failed to update global access');
            // Revert toggle
            event.target.checked = !allowed;
        }
    }

    async setPermission(personName, areaId, allowed, conditions = []) {
        const response = await fetch(`${this.apiBaseUrl}/segmentation/permission/add`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                person_name: personName,
                user_id: this.currentUserId,
                area_id: areaId,
                allowed: allowed,
                conditions: conditions.map(c => ({ value: c }))
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to set permission');
        }

        return await response.json();
    }

    async grantAllAccess() {
        if (!this.selectedPerson) return;

        try {
            await this.setPermission(this.selectedPerson, 'all', true, []);
            await this.loadExistingPermissions();
            this.displayPersonPermissions();
            this.updateSummary();
        } catch (error) {
            console.error('Error granting all access:', error);
            this.showError('Failed to grant all access');
        }
    }

    async revokeAllAccess() {
        if (!this.selectedPerson) return;

        try {
            // Remove global permission
            await this.removePermission(this.selectedPerson, 'all');
            
            // Remove individual permissions
            for (const area of this.verifiedAreas) {
                try {
                    await this.removePermission(this.selectedPerson, area.area_id);
                } catch (error) {
                    // Individual permission might not exist, continue
                }
            }

            await this.loadExistingPermissions();
            this.displayPersonPermissions();
            this.updateSummary();
        } catch (error) {
            console.error('Error revoking all access:', error);
            this.showError('Failed to revoke all access');
        }
    }

    async removePermission(personName, areaId) {
        const response = await fetch(`${this.apiBaseUrl}/segmentation/permission/person/${encodeURIComponent(personName)}/user/${this.currentUserId}/area/${areaId}`, {
            method: 'DELETE'
        });

        if (!response.ok && response.status !== 404) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to remove permission');
        }

        return response.ok ? await response.json() : null;
    }

    openPermissionModal(area) {
        this.currentModalArea = area;
        this.modalAreaType.textContent = area.area_type.replace('_', ' ');
        this.modalAreaSize.textContent = area.dimensions?.area?.toFixed(1) + 'm²' || 'N/A';
        this.modalPersonName.textContent = this.selectedPerson;

        const personPermissions = this.permissions.get(this.selectedPerson) || new Map();
        const permission = personPermissions.get(area.area_id);
        const hasAccess = permission?.allowed || false;
        const conditions = permission?.conditions || [];

        // Set current access status
        this.modalCurrentAccess.textContent = hasAccess ? 'Allowed' : 'Denied';

        // Set radio buttons
        const allowedRadio = document.querySelector('input[name="access-level"][value="allowed"]');
        const deniedRadio = document.querySelector('input[name="access-level"][value="denied"]');
        allowedRadio.checked = hasAccess;
        deniedRadio.checked = !hasAccess;

        // Show/hide conditions
        this.conditionsSection.style.display = hasAccess ? 'block' : 'none';

        // Set condition checkboxes
        document.querySelectorAll('input[name="condition"]').forEach(checkbox => {
            checkbox.checked = conditions.includes(checkbox.value);
        });

        this.modal.classList.remove('hidden');
    }

    closeModal() {
        this.modal.classList.add('hidden');
        this.currentModalArea = null;
    }

    async savePermission() {
        if (!this.currentModalArea || !this.selectedPerson) return;

        const allowedRadio = document.querySelector('input[name="access-level"][value="allowed"]');
        const allowed = allowedRadio.checked;

        const conditions = [];
        if (allowed) {
            document.querySelectorAll('input[name="condition"]:checked').forEach(checkbox => {
                conditions.push(checkbox.value);
            });
        }

        try {
            await this.setPermission(this.selectedPerson, this.currentModalArea.area_id, allowed, conditions);
            await this.loadExistingPermissions();
            this.displayPersonPermissions();
            this.updateSummary();
            this.closeModal();
        } catch (error) {
            console.error('Error saving permission:', error);
            this.showError('Failed to save permission');
        }
    }

    applyAdultPreset() {
        if (!this.selectedPerson) return;
        // Grant all access for adults
        this.grantAllAccess();
    }

    async applyChildPreset() {
        if (!this.selectedPerson) return;

        try {
            // Revoke all access first
            await this.revokeAllAccess();

            // Grant limited access with conditions
            const safeAreas = ['lawn', 'backyard', 'front_yard', 'patio', 'deck'];
            
            for (const area of this.verifiedAreas) {
                if (safeAreas.includes(area.area_type)) {
                    await this.setPermission(this.selectedPerson, area.area_id, true, ['adult_supervision_required', 'daylight_only']);
                }
            }

            await this.loadExistingPermissions();
            this.displayPersonPermissions();
            this.updateSummary();
        } catch (error) {
            console.error('Error applying child preset:', error);
            this.showError('Failed to apply child preset');
        }
    }

    async applyGuestPreset() {
        if (!this.selectedPerson) return;

        try {
            // Revoke all access first
            await this.revokeAllAccess();

            // Grant access to common areas only
            const commonAreas = ['front_yard', 'patio', 'lawn'];
            
            for (const area of this.verifiedAreas) {
                if (commonAreas.includes(area.area_type)) {
                    await this.setPermission(this.selectedPerson, area.area_id, true, []);
                }
            }

            await this.loadExistingPermissions();
            this.displayPersonPermissions();
            this.updateSummary();
        } catch (error) {
            console.error('Error applying guest preset:', error);
            this.showError('Failed to apply guest preset');
        }
    }

    updateSummary() {
        this.totalPeopleElement.textContent = this.labeledPeople.length;
        this.totalAreasElement.textContent = this.verifiedAreas.length;

        // Count total permissions
        let totalPermissions = 0;
        this.permissions.forEach(personPermissions => {
            totalPermissions += personPermissions.size;
        });
        this.totalPermissionsElement.textContent = totalPermissions;

        // Calculate completion rate
        const expectedPermissions = this.labeledPeople.length * this.verifiedAreas.length;
        const completionRate = expectedPermissions > 0 ? Math.round((totalPermissions / expectedPermissions) * 100) : 0;
        this.completionRateElement.textContent = completionRate + '%';

        // Enable finish button if at least some permissions are set
        this.finishSetupBtn.disabled = totalPermissions === 0;
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorState.classList.remove('hidden');
        setTimeout(() => {
            this.errorState.classList.add('hidden');
        }, 5000);
    }

    navigateToSegmentation() {
        const params = new URLSearchParams({
            video_path: this.currentVideoPath || '',
            user_id: this.currentUserId || '',
            service_url: this.apiBaseUrl
        });
        window.location.href = `../segmentation/index.html?${params.toString()}`;
    }

    finishSetup() {
        // Navigate to monitoring interface or show completion message
        alert(`Setup completed successfully!\n\nConfiguration Summary:\n- ${this.labeledPeople.length} people configured\n- ${this.verifiedAreas.length} areas defined\n- ${this.permissions.size} people with permissions\n\nAlan AI is now ready to monitor access violations.`);
        
        // Navigate to monitoring interface
        const params = new URLSearchParams({
            video_path: this.currentVideoPath || '',
            user_id: this.currentUserId || '',
            service_url: this.apiBaseUrl
        });
        window.location.href = `../monitoring/index.html?${params.toString()}`;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AccessPermissionsUI();
});