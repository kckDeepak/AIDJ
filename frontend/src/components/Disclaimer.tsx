/**
 * Global Disclaimer / Rights Notice Component
 * Appears at the bottom of every page
 */

'use client';

export function Disclaimer() {
    return (
        <div style={{
            width: '100%',
            padding: '16px 20px',
            background: 'var(--bg-surface)',
            border: '1px solid var(--danger)',
            borderRadius: '8px',
            marginTop: '24px',
        }}>
            <h4 style={{
                margin: '0 0 12px 0',
                fontSize: '13px',
                fontWeight: '600',
                color: 'var(--gray-100)',
                letterSpacing: '0.5px',
            }}>
                Private Test / Rights Notice
            </h4>
            
            <div style={{
                fontSize: '11px',
                lineHeight: '1.6',
                color: 'var(--gray-300)',
            }}>
                <p style={{ margin: '0 0 10px 0' }}>
                    This build is provided strictly for private testing and evaluation purposes only.
                </p>
                
                <p style={{ margin: '0 0 10px 0' }}>
                    This application does not include or grant any music reproduction, distribution, or public performance licences. Digital DJ-style processing may create temporary copies of audio (including but not limited to buffering, decoding, time-stretching, beatmatching, crossfading, or caching), which may require explicit permission from rights holders. Such permissions are not provided by standard DJ download or ownership licences.
                </p>
                
                <p style={{ margin: '0 0 10px 0' }}>
                    This software must not be used for public performances, public testing, commercial use, or public streaming unless an approved licensing agreement is confirmed (for example, through an authorised streaming or licensing partner) and the user holds all necessary rights and permissions.
                </p>
                
                <p style={{ margin: '0' }}>
                    By continuing to use this application, you acknowledge that you are responsible for ensuring compliance with all applicable copyright and licensing requirements.
                </p>
            </div>
        </div>
    );
}
