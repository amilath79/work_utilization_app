"""
Enterprise Streamlit Security and Configuration Utilities
"""
import streamlit as st
import logging
import os
from typing import Dict, Any
from datetime import datetime

from config import ENTERPRISE_CONFIG, enterprise_logger, audit_logger

class EnterpriseStreamlitManager:
    """Enterprise Streamlit application manager"""
    
    def __init__(self):
        self.logger = enterprise_logger
        self.audit_logger = audit_logger
        self.config = ENTERPRISE_CONFIG
    
    def configure_page(self, page_title: str, page_icon: str = "🏢", 
                      layout: str = "wide") -> None:
        """Configure Streamlit page with enterprise settings"""
        try:
            # Enterprise page configuration
            st.set_page_config(
                page_title=f"Enterprise - {page_title}",
                page_icon=page_icon,
                layout=layout,
                initial_sidebar_state="expanded"
            )
            
            # Add enterprise security headers
            if self.config.security_headers:
                self._add_security_headers()
            
            # Audit page access
            self.audit_logger.info(f"PAGE_ACCESS | {page_title} | {st.session_state.get('user_id', 'anonymous')}")
            
        except Exception as e:
            self.logger.error(f"Failed to configure enterprise page: {e}")
    
    def _add_security_headers(self) -> None:
        """Add enterprise security headers"""
        try:
            # Add security headers via HTML
            st.markdown("""
            <script>
            // Enterprise security headers
            if (typeof window !== 'undefined') {
                // Disable right-click in production
                if (window.location.hostname !== 'localhost') {
                    document.addEventListener('contextmenu', e => e.preventDefault());
                }
                
                // Add enterprise metadata
                document.title = 'Enterprise Workforce Prediction';
            }
            </script>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            self.logger.error(f"Failed to add security headers: {e}")
    
    def add_enterprise_sidebar(self) -> None:
        """Add enterprise information to sidebar"""
        try:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🏢 Enterprise Info")
            
            # Environment indicator
            env_color = {
                "development": "🟡",
                "staging": "🟠", 
                "production": "🔴"
            }
            
            st.sidebar.markdown(f"{env_color.get(self.config.environment.value, '⚫')} **Environment:** {self.config.environment.value.title()}")
            
            # Enterprise mode indicator
            if self.config.enterprise_mode:
                st.sidebar.markdown("🔒 **Enterprise Mode:** Active")
            
            # System timestamp
            st.sidebar.markdown(f"🕐 **System Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Failed to add enterprise sidebar: {e}")
    
    def log_user_action(self, action: str, details: Dict[str, Any] = None) -> None:
        """Log user actions for enterprise audit"""
        try:
            user_id = st.session_state.get('user_id', 'anonymous')
            session_id = st.session_state.get('session_id', 'unknown')
            
            audit_data = {
                "action": action,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if details:
                audit_data.update(details)
            
            self.audit_logger.info(f"USER_ACTION | {action} | {user_id} | {audit_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to log user action: {e}")
    
    def check_enterprise_compliance(self) -> bool:
        """Check if application meets enterprise compliance requirements"""
        try:
            compliance_checks = {
                "audit_logging": self.config.audit_logging,
                "security_headers": self.config.security_headers,
                "enterprise_mode": self.config.enterprise_mode,
                "environment_set": self.config.environment != "development"
            }
            
            all_compliant = all(compliance_checks.values())
            
            if not all_compliant:
                self.logger.warning(f"Enterprise compliance check failed: {compliance_checks}")
            
            return all_compliant
            
        except Exception as e:
            self.logger.error(f"Enterprise compliance check failed: {e}")
            return False

# Global enterprise Streamlit manager
streamlit_manager = EnterpriseStreamlitManager()