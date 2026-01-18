const API_BASE = 'http://127.0.0.1:8000';

const getStoredUser = () => {
    const userStr = localStorage.getItem('sao_user');
    return userStr ? JSON.parse(userStr) : null;
};

const updateLandingUI = () => {
    const token = localStorage.getItem('sao_token');
    const user = getStoredUser();
    
    if (token && user) {
        const nav = document.querySelector('header nav');
        nav.innerHTML = `
            <span style="margin-right: 15px; color: #9aa8b8;">Hello, ${user.full_name || 'User'}</span>
            <button id="getStartedBtn" style="background: #00d4ff; color: #04202a; padding: 10px 14px; border-radius: 6px; border: none; font-weight: 700; cursor: pointer;">Open App</button>
            <button id="logoutBtn" style="background: transparent; color: #9aa8b8; padding: 10px 14px; border-radius: 6px; border: 1px solid #233246; cursor: pointer; margin-left: 8px;">Logout</button>
        `;
        
        document.getElementById('getStartedBtn').addEventListener('click', () => {
            window.location.href = 'index.html';
        });
        
        document.getElementById('logoutBtn').addEventListener('click', () => {
            localStorage.removeItem('sao_token');
            localStorage.removeItem('sao_user');
            window.location.href = 'landing.html';
        });
    }
};

document.addEventListener('DOMContentLoaded', updateLandingUI);
