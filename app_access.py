"""Optional OIDC access control for a shared, trusted-team deployment."""
import os
import streamlit as st


def require_access():
    mode = os.getenv('RAG_REQUIRE_AUTH', 'false').strip().lower()
    if mode in ('false', '0'):
        return
    if mode not in ('true', '1'):
        st.error('Invalid authentication configuration. Set RAG_REQUIRE_AUTH to true or false.')
        st.stop()
    allowed = {email.strip().casefold() for email in os.getenv('RAG_ALLOWED_EMAILS', '').split(',') if email.strip()}
    if not allowed:
        st.error('Access is locked. Configure RAG_ALLOWED_EMAILS on the server.')
        st.stop()
    if not st.user.is_logged_in:
        st.title('Welcome to Paper Desk')
        st.write('Sign in to access your team’s research collection.')
        if st.button('Sign in', type='primary'):
            try:
                st.login()
            except Exception:
                st.error('Sign-in is unavailable. Check the server’s identity-provider configuration.')
        st.stop()
    email = str(st.user.get('email', '')).casefold()
    if st.user.get('email_verified') is not True or email not in allowed:
        st.error('This account does not have access to the workspace.')
        st.button('Sign out', on_click=st.logout)
        st.stop()
    with st.sidebar:
        st.caption('Signed in as ' + email)
        st.button('Sign out', on_click=st.logout)
