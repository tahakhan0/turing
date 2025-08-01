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
        this.peopleList = document.getElementById('people-list');
        this.emptyPermissions = document.getElementById('empty-permissions');

        // Permissions configuration
        this.permissionsConfig = document.getElementById('permissions-config');
        this.selectedPersonName = document.getElementById('selected-person-name');
        this.grantAllAccessBtn = document.getElementById('grant-all-access');
        this.revokeAllAccessBtn = document.getElementById('revoke-all-access');
        this.globalAccessToggle = document.getElementById('global-access-toggle');
        this.areaPermissionsGrid = document.getElementById('area-permissions-grid');

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
        // Load data with individual error handling to prevent one failure from blocking the page
        try {
            await this.checkServiceConnection();
        } catch (error) {
            console.error('Service connection check failed:', error);
            // Continue loading even if connection check fails
        }

        try {
            await this.loadLabeledPeople();
        } catch (error) {
            console.error('Failed to load labeled people:', error);
            this.labeledPeople = []; // Fallback to empty array
        }

        try {
            await this.loadVerifiedAreas();
        } catch (error) {
            console.error('Failed to load verified areas:', error);
            this.verifiedAreas = []; // Fallback to empty array
        }

        // Always display the interface, even with partial data
        this.displayInterface();
        this.updateSummary();
        this.loadingState.classList.add('hidden');

        // Show error if no data was loaded
        if (this.labeledPeople.length === 0 && this.verifiedAreas.length === 0) {
            this.showError('Could not load permission data. Please check your connection and try refreshing.');
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
        console.log(`Loading labeled people from: ${this.apiBaseUrl}/segmentation/people/${this.currentUserId}`);
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch(`${this.apiBaseUrl}/segmentation/people/${this.currentUserId}`, {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`Failed to load labeled people: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.labeledPeople = data.people || [];
            console.log('Loaded labeled people:', this.labeledPeople);
            
        } catch (error) {
            console.error('Error loading labeled people:', error);
            if (error.name === 'AbortError') {
                throw new Error('Timeout loading labeled people from face recognition data');
            }
            throw new Error('Could not load labeled people from face recognition data');
        }
    }

    async loadVerifiedAreas() {
        console.log(`Loading verified areas from: ${this.apiBaseUrl}/segmentation/user/${this.currentUserId}`);
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch(`${this.apiBaseUrl}/segmentation/user/${this.currentUserId}`, {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`Failed to load areas: ${response.statusText}`);
            }
            
            const data = await response.json();
            // Get areas from segmentation_data.segments since that's the structure
            const segments = data.segmentation_data?.segments || [];
            this.verifiedAreas = segments; // All segments are considered verified for permissions
            console.log('Loaded verified areas:', this.verifiedAreas);
            
        } catch (error) {
            console.error('Error loading verified areas:', error);
            if (error.name === 'AbortError') {
                throw new Error('Timeout loading verified areas from segmentation data');
            }
            throw new Error('Could not load verified areas from segmentation data');
        }
    }

    async loadPersonPermissions(person) {
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
            } else {
                // If no existing permissions found, initialize with empty permissions
                this.permissions.set(person, new Map());
            }
        } catch (error) {
            console.error(`Error loading permissions for ${person}:`, error);
            // Initialize with empty permissions on error
            this.permissions.set(person, new Map());
        }
    }

    // Legacy method - now simplified for batch operations if needed
    async loadExistingPermissions() {
        // This method is now only used for batch operations if needed
        // Individual permissions are loaded per person via loadPersonPermissions
        console.log('Batch permission loading - use loadPersonPermissions for better performance');
    }

    displayInterface() {
        this.permissionsInterface.classList.remove('hidden');
        this.displayPeopleList();
    }

    displayPeopleList() {
        this.peopleList.innerHTML = '';

        this.labeledPeople.forEach(person => {
            const personItem = this.createPersonListItem(person);
            this.peopleList.appendChild(personItem);
        });
    }

    createPersonListItem(person) {
        const item = document.createElement('div');
        item.className = `person-list-item p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-all`;
        
        item.innerHTML = `
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 bg-brand-purple rounded-full flex items-center justify-center flex-shrink-0">
                    <span class="text-white text-sm font-bold">${person.charAt(0).toUpperCase()}</span>
                </div>
                <div class="flex-1 min-w-0">
                    <p class="font-medium text-gray-900 truncate">${person}</p>
                    <div class="flex items-center space-x-2 mt-1">
                        <div class="w-2 h-2 bg-gray-300 rounded-full"></div>
                        <span class="text-xs text-gray-500">Not configured</span>
                    </div>
                </div>
            </div>
        `;

        item.addEventListener('click', () => this.selectPerson(person, item));
        return item;
    }

    async selectPerson(person, itemElement) {
        // Update UI selection
        document.querySelectorAll('.person-list-item').forEach(item => {
            item.classList.remove('selected');
        });
        itemElement.classList.add('selected');

        this.selectedPerson = person;
        this.selectedPersonName.textContent = person;
        
        // Hide empty state and show permissions config
        this.emptyPermissions.classList.add('hidden');
        this.permissionsConfig.classList.remove('hidden');

        // Load permissions for this specific person only when selected
        await this.loadPersonPermissions(person);
        this.displayPersonPermissions();
        this.updatePersonStatus(person, itemElement);
    }

    displayPersonPermissions() {
        const personPermissions = this.permissions.get(this.selectedPerson) || new Map();
        
        // Update global access toggle
        const hasGlobalAccess = personPermissions.has('all') && personPermissions.get('all').allowed;
        this.globalAccessToggle.checked = hasGlobalAccess;

        // Display individual area permissions using Set to avoid duplicates
        this.areaPermissionsGrid.innerHTML = '';
        
        // Use Set to get unique area types
        const uniqueAreaTypes = new Set();
        const uniqueAreas = [];
        
        this.verifiedAreas.forEach(area => {
            if (!uniqueAreaTypes.has(area.area_type)) {
                uniqueAreaTypes.add(area.area_type);
                uniqueAreas.push(area);
            }
        });

        uniqueAreas.forEach(area => {
            const permissionCard = this.createAreaPermissionCard(area, personPermissions);
            this.areaPermissionsGrid.appendChild(permissionCard);
        });
    }

    createAreaPermissionCard(area, personPermissions) {
        const permission = personPermissions.get(area.area_id);
        const hasAccess = permission?.allowed || false;
        const conditions = permission?.conditions || [];
        const isGlobalAccess = personPermissions.has('all') && personPermissions.get('all').allowed;

        const card = document.createElement('div');
        card.className = 'border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow';
        
        const statusColor = (hasAccess || isGlobalAccess) ? 'text-green-600' : 'text-gray-400';
        const statusIcon = (hasAccess || isGlobalAccess) ? '✓' : '○';
        
        card.innerHTML = `
            <div class="flex items-center justify-between mb-3">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-blue-100 to-purple-100 rounded-lg flex items-center justify-center">
                        <span class="text-lg">${this.getAreaIcon(area.area_type)}</span>
                    </div>
                    <div>
                        <h4 class="font-medium text-gray-900 capitalize">${area.area_type.replace('_', ' ')}</h4>
                        <div class="flex items-center space-x-2 text-xs">
                            <span class="${statusColor} font-medium">${statusIcon} ${(hasAccess || isGlobalAccess) ? 'Allowed' : 'Denied'}</span>
                            ${conditions.length > 0 ? `<span class="text-orange-600">• With conditions</span>` : ''}
                            ${isGlobalAccess ? `<span class="text-blue-600">• Via global access</span>` : ''}
                        </div>
                    </div>
                </div>
                <button class="config-btn bg-gray-100 hover:bg-gray-200 p-2 rounded-lg transition-colors" 
                        data-area-id="${area.area_id}">
                    <svg class="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                    </svg>
                </button>
            </div>
            <div class="flex items-center justify-between">
                <span class="text-sm text-gray-600">Access Control</span>
                <label class="toggle-switch">
                    <input type="checkbox" ${(hasAccess || isGlobalAccess) ? 'checked' : ''} ${isGlobalAccess ? 'disabled' : ''} 
                           data-area-id="${area.area_id}" class="area-toggle">
                    <span class="toggle-slider"></span>
                </label>
            </div>
        `;

        // Add event listeners
        const toggle = card.querySelector('.area-toggle');
        const configBtn = card.querySelector('.config-btn');

        toggle.addEventListener('change', (e) => this.handleAreaToggle(e));
        configBtn.addEventListener('click', (e) => this.openPermissionModal(area));

        return card;
    }

    updatePersonStatus(person, itemElement) {
        const personPermissions = this.permissions.get(person) || new Map();
        const hasPermissions = personPermissions.size > 0;
        const statusDot = itemElement.querySelector('.w-2');
        const statusText = itemElement.querySelector('.text-xs');
        
        if (hasPermissions) {
            statusDot.className = 'w-2 h-2 bg-green-400 rounded-full';
            statusText.textContent = 'Configured';
            statusText.className = 'text-xs text-green-600';
        } else {
            statusDot.className = 'w-2 h-2 bg-gray-300 rounded-full';
            statusText.textContent = 'Not configured';
            statusText.className = 'text-xs text-gray-500';
        }
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
        
        // Find the area name for the toast message
        const area = this.verifiedAreas.find(a => a.area_id === areaId);
        const areaName = area ? area.area_type.replace('_', ' ') : 'area';

        try {
            await this.setPermission(this.selectedPerson, areaId, allowed, []);
            await this.loadPersonPermissions(this.selectedPerson);
            this.displayPersonPermissions();
            this.updateSummary();
            
            // Update person status in sidebar
            const selectedItem = document.querySelector('.person-list-item.selected');
            if (selectedItem) {
                this.updatePersonStatus(this.selectedPerson, selectedItem);
            }
            
            // Show success toast
            const action = allowed ? 'granted' : 'revoked';
            this.showToast(`${this.selectedPerson} has been ${action} access to ${areaName}`);
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
            await this.loadPersonPermissions(this.selectedPerson);
            this.displayPersonPermissions();
            this.updateSummary();
            
            // Update person status in sidebar
            const selectedItem = document.querySelector('.person-list-item.selected');
            if (selectedItem) {
                this.updatePersonStatus(this.selectedPerson, selectedItem);
            }
            
            // Show success toast
            const action = allowed ? 'granted full property access' : 'revoked full property access';
            this.showToast(`${this.selectedPerson} has been ${action}`);
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
            await this.loadPersonPermissions(this.selectedPerson);
            this.displayPersonPermissions();
            this.updateSummary();
            
            // Update person status in sidebar
            const selectedItem = document.querySelector('.person-list-item.selected');
            if (selectedItem) {
                this.updatePersonStatus(this.selectedPerson, selectedItem);
            }
            
            // Show success toast
            this.showToast(`${this.selectedPerson} has been granted access to all areas`);
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

            await this.loadPersonPermissions(this.selectedPerson);
            this.displayPersonPermissions();
            this.updateSummary();
            
            // Update person status in sidebar
            const selectedItem = document.querySelector('.person-list-item.selected');
            if (selectedItem) {
                this.updatePersonStatus(this.selectedPerson, selectedItem);
            }
            
            // Show success toast
            this.showToast(`All access permissions revoked for ${this.selectedPerson}`);
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
            await this.loadPersonPermissions(this.selectedPerson);
            this.displayPersonPermissions();
            this.updateSummary();
            this.closeModal();
            
            // Update person status in sidebar
            const selectedItem = document.querySelector('.person-list-item.selected');
            if (selectedItem) {
                this.updatePersonStatus(this.selectedPerson, selectedItem);
            }
            
            // Show success toast
            const areaName = this.currentModalArea.area_type.replace('_', ' ');
            const action = allowed ? 'granted' : 'denied';
            this.showToast(`${this.selectedPerson} has been ${action} access to ${areaName}`);
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

            await this.loadPersonPermissions(this.selectedPerson);
            this.displayPersonPermissions();
            this.updateSummary();
            
            // Update person status in sidebar
            const selectedItem = document.querySelector('.person-list-item.selected');
            if (selectedItem) {
                this.updatePersonStatus(this.selectedPerson, selectedItem);
            }
            
            // Show success toast
            this.showToast(`Child safety preset applied for ${this.selectedPerson}`);
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

            await this.loadPersonPermissions(this.selectedPerson);
            this.displayPersonPermissions();
            this.updateSummary();
            
            // Update person status in sidebar
            const selectedItem = document.querySelector('.person-list-item.selected');
            if (selectedItem) {
                this.updatePersonStatus(this.selectedPerson, selectedItem);
            }
            
            // Show success toast
            this.showToast(`Guest access preset applied for ${this.selectedPerson}`);
        } catch (error) {
            console.error('Error applying guest preset:', error);
            this.showError('Failed to apply guest preset');
        }
    }

    updateSummary() {
        this.totalPeopleElement.textContent = this.labeledPeople.length;
        this.totalAreasElement.textContent = this.verifiedAreas.length;

        // Count permissions for currently selected person only (for better performance)
        const selectedPersonPermissions = this.selectedPerson ? 
            (this.permissions.get(this.selectedPerson)?.size || 0) : 0;
        this.totalPermissionsElement.textContent = selectedPersonPermissions;

        // Simplified completion - show "Ready" when we have people and areas
        const isReadyToDefinePermissions = this.labeledPeople.length > 0 && this.verifiedAreas.length > 0;
        this.completionRateElement.textContent = isReadyToDefinePermissions ? 'Ready' : '0%';

        // Enable finish button when we have basic setup (people and areas)
        this.finishSetupBtn.disabled = !isReadyToDefinePermissions;
    }

    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        const bgColor = type === 'success' ? 'bg-green-600' : 'bg-red-600';
        const icon = type === 'success' 
            ? '<svg class="w-5 h-5 text-white mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>'
            : '<svg class="w-5 h-5 text-white mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path></svg>';
        
        toast.className = `fixed bottom-5 right-5 text-white px-6 py-3 rounded-lg shadow-lg transform translate-y-20 opacity-0 transition-all duration-300 ease-out ${bgColor} z-50`;
        toast.innerHTML = `
            <div class="flex items-center">
                ${icon}
                <span>${message}</span>
            </div>
        `;

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
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            }, 300);
        }, 4000);
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorState.classList.remove('hidden');
        setTimeout(() => {
            this.errorState.classList.add('hidden');
        }, 5000);
        
        // Also show error toast
        this.showToast(message, 'error');
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
        // Count total permissions across all people
        let totalPermissions = 0;
        this.permissions.forEach(personPermissions => {
            totalPermissions += personPermissions.size;
        });
        
        // Navigate to completion page with summary stats
        const params = new URLSearchParams({
            video_path: this.currentVideoPath || '',
            user_id: this.currentUserId || '',
            service_url: this.apiBaseUrl,
            people_count: this.labeledPeople.length.toString(),
            areas_count: this.verifiedAreas.length.toString(),
            permissions_count: totalPermissions.toString()
        });
        window.location.href = `../completion/index.html?${params.toString()}`;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AccessPermissionsUI();
});