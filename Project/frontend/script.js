const API_BASE = 'http://127.0.0.1:8000';

let currentChatId = null;
let currentUser = null;
let chatHistory = [];

const getAuthHeaders = () => {
    const token = localStorage.getItem('sao_token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
};

const getStoredUser = () => {
    const userStr = localStorage.getItem('sao_user');
    return userStr ? JSON.parse(userStr) : null;
};

const showToast = (message, type = 'info') => {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 3000);
};

const handleApiError = async (response) => {
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'An error occurred' }));
        throw new Error(error.detail || 'An error occurred');
    }
    return response.json();
};

const initApp = async () => {
    const token = localStorage.getItem('sao_token');
    console.log('Token in localStorage:', token ? 'Found' : 'Not found');
    if (!token) {
        window.location.href = 'signin.html';
        return;
    }

    currentUser = getStoredUser();
    console.log('User from localStorage:', currentUser);
    if (!currentUser) {
        console.log('No user found, logging out...');
        logout();
        return;
    }

    updateUserUI();
    await loadChatHistory();
    setupEventListeners();
};

const updateUserUI = () => {
    if (currentUser) {
        const initials = currentUser.full_name ? currentUser.full_name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) : '?';
        document.getElementById('userName').textContent = currentUser.full_name || 'User';
        document.getElementById('userAvatar').textContent = initials;
        document.getElementById('profileAvatar').textContent = initials;
    }
};

const setupEventListeners = () => {
    document.getElementById('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    document.getElementById('chatInput').addEventListener('input', (e) => {
        e.target.style.height = 'auto';
        e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px';
    });

    document.getElementById('newChatBtn').addEventListener('click', startNewChat);
    document.getElementById('collapseBtn').addEventListener('click', toggleSidebar);
};

const toggleSidebar = () => {
    document.getElementById('sidebar').classList.toggle('closed');
};

const startNewChat = async () => {
    currentChatId = null;
    document.getElementById('messages').innerHTML = `
        <div class="msg assistant">
            <div class="msg-content">
                <p>Hello! I'm SAO Health Assistant. Describe your symptoms, and I'll help predict potential conditions and provide recommendations.</p>
                <p>For example: "I have fever, cough, and headache"</p>
            </div>
        </div>
    `;
    document.getElementById('currentChatTitle').textContent = 'New chat';
};

const loadChatHistory = async () => {
    try {
        const response = await fetch(`${API_BASE}/chats`, {
            headers: getAuthHeaders()
        });
        if (response.ok) {
            chatHistory = await response.json();
            renderChatHistory();
        }
    } catch (error) {
        console.error('Failed to load chat history:', error);
    }
};

const renderChatHistory = () => {
    const chatList = document.getElementById('chatList');
    chatList.innerHTML = '';

    if (chatHistory.length === 0) {
        chatList.innerHTML = '<div style="text-align: center; color: var(--muted); padding: 20px; font-size: 13px;">No previous chats</div>';
        return;
    }

    chatHistory.forEach(chat => {
        const item = document.createElement('div');
        item.className = `chat-item${currentChatId === chat.id ? ' active' : ''}`;
        item.onclick = () => loadChat(chat.id);
        item.innerHTML = `
            <span class="chat-item-icon">💬</span>
            <span class="chat-item-title">${chat.title || 'New Chat'}</span>
        `;
        chatList.appendChild(item);
    });
};

const loadChat = async (chatId) => {
    try {
        currentChatId = chatId;
        const response = await fetch(`${API_BASE}/chats/${chatId}`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) throw new Error('Failed to load chat');

        const data = await response.json();
        const messages = data.messages;

        document.getElementById('messages').innerHTML = '';

        if (messages.length === 0) {
            document.getElementById('messages').innerHTML = `
                <div class="msg assistant">
                    <div class="msg-content">
                        <p>Hello! I'm SAO Health Assistant. Describe your symptoms, and I'll help predict potential conditions and provide recommendations.</p>
                    </div>
                </div>
            `;
        } else {
            messages.forEach(msg => appendMessage(msg.role, msg.content));
        }

        document.getElementById('currentChatTitle').textContent = data.chat.title || 'New chat';
        renderChatHistory();
        scrollToBottom();

    } catch (error) {
        showToast('Failed to load chat', 'error');
        console.error(error);
    }
};

const appendMessage = (role, content, isPrediction = false) => {
    const messagesDiv = document.getElementById('messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}`;

    let messageContent = content;
    if (isPrediction && typeof content === 'object') {
        messageContent = formatPredictionContent(content);
    }

    msgDiv.innerHTML = `<div class="msg-content">${messageContent}</div>`;
    messagesDiv.appendChild(msgDiv);
    return msgDiv;
};

const formatPredictionContent = (prediction) => {
    let html = `<p>Based on your symptoms, I predict: <strong style="color: var(--accent); font-size: 18px;">${prediction.predicted_disease}</strong></p>`;

    if (prediction.probability !== null) {
        html += `<p class="probability">Confidence: ${(prediction.probability * 100).toFixed(1)}%</p>`;
    }

    if (prediction.matched_symptoms && prediction.matched_symptoms.length > 0) {
        html += `<p>Matched symptoms: ${prediction.matched_symptoms.join(', ')}</p>`;
    }

    // Display description if available
    if (prediction.description && prediction.description !== "No description found") {
        html += `<div class="disease-description"><h5>Description:</h5><p>${prediction.description}</p></div>`;
    }

    html += `<div class="precautions"><h5>Precautions:</h5><ul>`;
    if (prediction.precautions && prediction.precautions.length > 0) {
        prediction.precautions.forEach(p => {
            if (p && p.trim()) html += `<li>${p}</li>`;
        });
    } else {
        html += '<li>No specific precautions available</li>';
    }
    html += '</ul></div>';

    return `<div class="prediction-result">${html}</div>`;
};

const sendMessage = async () => {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) return;

    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;

    appendMessage('user', message);
    input.value = '';
    input.style.height = 'auto';

    const typingIndicator = document.getElementById('typingIndicator');
    typingIndicator.classList.remove('hidden');
    scrollToBottom();

    try {
        if (!currentChatId) {
            const createResponse = await fetch(`${API_BASE}/chats`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeaders()
                },
                body: JSON.stringify({ title: message.slice(0, 50) + (message.length > 50 ? '...' : '') })
            });

            if (!createResponse.ok) throw new Error('Failed to create chat');
            const chatData = await createResponse.json();
            currentChatId = chatData.id;
            document.getElementById('currentChatTitle').textContent = chatData.title || 'New chat';
            await loadChatHistory();
        }

        const response = await fetch(`${API_BASE}/predict_text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify({ user_input: message })
        });

        typingIndicator.classList.add('hidden');

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Prediction failed' }));
            throw new Error(error.detail || 'Prediction failed');
        }

        const prediction = await response.json();

        // Fetch disease details (description and precautions)
        try {
            const detailsResponse = await fetch(
                `${API_BASE}/get_details?disease=${encodeURIComponent(prediction.predicted_disease)}`,
                { headers: getAuthHeaders() }
            );
            if (detailsResponse.ok) {
                const details = await detailsResponse.json();
                prediction.description = details.description;
                prediction.precautions = details.precautions;
            }
        } catch (error) {
            console.warn('Failed to fetch disease details:', error);
            // Continue with prediction data that may not have full details
        }

        appendMessage('assistant', prediction, true);

        if (currentChatId) {
            await fetch(`${API_BASE}/chats/${currentChatId}/messages`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeaders()
                },
                body: JSON.stringify({ role: 'user', content: message })
            });

            await fetch(`${API_BASE}/chats/${currentChatId}/messages`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeaders()
                },
                body: JSON.stringify({ role: 'assistant', content: JSON.stringify(prediction) })
            });
        }

        scrollToBottom();

    } catch (error) {
        typingIndicator.classList.add('hidden');
        appendMessage('assistant', `I apologize, but I encountered an error: ${error.message}. Please try again.`);
        showToast(error.message, 'error');
        console.error('Prediction error:', error);
    } finally {
        sendBtn.disabled = false;
    }
};

const scrollToBottom = () => {
    const chatWindow = document.getElementById('chatWindow');
    chatWindow.scrollTop = chatWindow.scrollHeight;
};

const logout = () => {
    localStorage.removeItem('sao_token');
    localStorage.removeItem('sao_user');
    window.location.href = 'landing.html';
};

const showProfileModal = async () => {
    try {
        const response = await fetch(`${API_BASE}/user/profile`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) throw new Error('Failed to load profile');

        const user = await response.json();
        document.getElementById('profileName').value = user.full_name || '';
        document.getElementById('profileEmail').value = user.email || '';
        document.getElementById('profileDob').value = user.dob || '';
        document.getElementById('profileGender').value = user.gender || '';
        document.getElementById('profileNationality').value = user.nationality || '';

        const statsResponse = await fetch(`${API_BASE}/user/chat-stats`, {
            headers: getAuthHeaders()
        });
        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            document.getElementById('statChats').textContent = stats.total_chats || 0;
            document.getElementById('statMessages').textContent = stats.total_messages || 0;
        }

        document.getElementById('profileModal').classList.remove('hidden');

    } catch (error) {
        showToast('Failed to load profile', 'error');
        console.error(error);
    }
};

const closeProfileModal = () => {
    document.getElementById('profileModal').classList.add('hidden');
};

const updateProfile = async (event) => {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = {
        fullName: formData.get('fullName') || undefined,
        dob: formData.get('dob') || undefined,
        gender: formData.get('gender') || undefined,
        nationality: formData.get('nationality') || undefined
    };

    try {
        const response = await fetch(`${API_BASE}/user/profile`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Update failed' }));
            throw new Error(error.detail || 'Update failed');
        }

        const updatedUser = await response.json();
        currentUser = { ...currentUser, ...updatedUser };
        localStorage.setItem('sao_user', JSON.stringify(currentUser));
        updateUserUI();
        showToast('Profile updated successfully', 'success');
        closeProfileModal();

    } catch (error) {
        showToast(error.message, 'error');
    }
};

const showPasswordModal = () => {
    closeProfileModal();
    document.getElementById('passwordModal').classList.remove('hidden');
};

const closePasswordModal = () => {
    document.getElementById('passwordModal').classList.add('hidden');
    document.getElementById('passwordForm').reset();
};

const changePassword = async (event) => {
    event.preventDefault();

    const newPassword = document.getElementById('newPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const currentPassword = document.getElementById('currentPassword').value;

    if (newPassword !== confirmPassword) {
        showToast('New passwords do not match', 'error');
        return;
    }

    if (newPassword.length < 8) {
        showToast('Password must be at least 8 characters', 'error');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('current_password', currentPassword);
        formData.append('new_password', newPassword);

        const response = await fetch(`${API_BASE}/user/change-password`, {
            method: 'POST',
            headers: {
                ...getAuthHeaders()
            },
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Password change failed' }));
            throw new Error(error.detail || 'Password change failed');
        }

        showToast('Password changed successfully', 'success');
        closePasswordModal();

    } catch (error) {
        showToast(error.message, 'error');
    }
};

document.addEventListener('DOMContentLoaded', initApp);
