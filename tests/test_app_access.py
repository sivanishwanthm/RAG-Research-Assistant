import unittest
from unittest.mock import patch, MagicMock
from app_access import require_access

class Stopped(Exception): pass

class AccessTests(unittest.TestCase):
    def check_blocked(self, environ, user):
        with patch.dict('os.environ', environ, clear=True), patch('app_access.st') as st:
            st.stop.side_effect = Stopped
            st.user = user
            st.button.return_value = False
            with self.assertRaises(Stopped): require_access()
            st.stop.assert_called_once()

    def test_local_default_does_not_require_identity(self):
        with patch.dict('os.environ', {}, clear=True), patch('app_access.st') as st:
            require_access()
            st.stop.assert_not_called()

    def test_auth_fails_closed_without_allowlist(self):
        self.check_blocked({'RAG_REQUIRE_AUTH':'true'}, MagicMock())

    def test_unverified_identity_is_rejected(self):
        user = MagicMock(is_logged_in=True)
        user.get.side_effect = lambda k, default=None: {'email':'owner@example.test', 'email_verified':False}.get(k,default)
        self.check_blocked({'RAG_REQUIRE_AUTH':'true', 'RAG_ALLOWED_EMAILS':'owner@example.test'}, user)

    def test_verified_allowed_identity(self):
        user = MagicMock(is_logged_in=True)
        user.get.side_effect = lambda k, default=None: {'email':'owner@example.test', 'email_verified':True}.get(k,default)
        with patch.dict('os.environ', {'RAG_REQUIRE_AUTH':'true', 'RAG_ALLOWED_EMAILS':'owner@example.test'}, clear=True), patch('app_access.st') as st:
            st.user = user
            require_access()
            st.stop.assert_not_called()
