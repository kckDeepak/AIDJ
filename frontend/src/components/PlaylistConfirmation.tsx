/**
 * Playlist Confirmation Component
 * ================================
 * Shows generated playlist for user confirmation before mixing.
 * Allows reordering, removing songs, and song requests.
 */

'use client';

import { useState, useEffect } from 'react';
import { GripVertical, X, ArrowRight, Music, Sparkles, Play, CheckCircle } from 'lucide-react';

interface Song {
  id: string;
  name: string;
  bpm: number;
  key: string;
  file: File;
  isGenerated?: boolean;
  duration?: number;
}

interface PlaylistConfirmationProps {
  playlist: Song[];
  occasion: string;
  onConfirm: (playlist: Song[]) => void;
  onCancel: () => void;
  onSongRequest: (songId: string, nearSongId: string) => void;
  onRemoveSong: (songId: string) => void;
  onReorderPlaylist: (fromIndex: number, toIndex: number) => void;
}

export function PlaylistConfirmation({
  playlist,
  occasion,
  onConfirm,
  onCancel,
  onSongRequest,
  onRemoveSong,
  onReorderPlaylist,
}: PlaylistConfirmationProps) {
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const [showSongRequestModal, setShowSongRequestModal] = useState(false);
  const [selectedSongForRequest, setSelectedSongForRequest] = useState<string | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth <= 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    setDragOverIndex(index);
  };

  const handleDrop = (index: number) => {
    if (draggedIndex !== null && draggedIndex !== index) {
      onReorderPlaylist(draggedIndex, index);
    }
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const handleSongRequest = (songId: string) => {
    setSelectedSongForRequest(songId);
    setShowSongRequestModal(true);
  };

  const handleConfirmSongRequest = (nearSongId: string) => {
    if (selectedSongForRequest) {
      onSongRequest(selectedSongForRequest, nearSongId);
    }
    setShowSongRequestModal(false);
    setSelectedSongForRequest(null);
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.8)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 2000,
      padding: isMobile ? '12px' : '20px',
    }}>
      <div style={{
        width: '100%',
        maxWidth: isMobile ? '100%' : '700px',
        maxHeight: isMobile ? '95vh' : '85vh',
        background: 'rgba(20, 20, 30, 0.95)',
        border: '1px solid rgba(74, 158, 255, 0.3)',
        borderRadius: isMobile ? '16px' : '24px',
        boxShadow: '0 25px 100px rgba(0, 0, 0, 0.5), 0 0 60px rgba(74, 158, 255, 0.2)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          padding: isMobile ? '16px 16px 14px' : '24px 28px 20px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          background: 'linear-gradient(180deg, rgba(74, 158, 255, 0.1), transparent)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
            <Sparkles size={isMobile ? 20 : 24} color="var(--accent)" />
            <h2 style={{ margin: 0, fontSize: isMobile ? '18px' : '22px', fontWeight: '700' }}>
              AI Generated Playlist
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: isMobile ? '13px' : '14px', color: '#888' }}>
            For: <span style={{ color: 'var(--accent)', fontWeight: '500' }}>{occasion}</span>
          </p>
          {!isMobile && (
            <p style={{ margin: '8px 0 0', fontSize: '13px', color: '#666' }}>
              Drag to reorder • Click × to remove • Click song to request placement
            </p>
          )}
        </div>

        {/* Playlist */}
        <div style={{
          flex: 1,
          overflow: 'auto',
          padding: isMobile ? '12px 12px' : '16px 20px',
        }}>
          {playlist.map((song, index) => (
            <div
              key={song.id}
              draggable={!isMobile}
              onDragStart={() => !isMobile && handleDragStart(index)}
              onDragOver={(e) => !isMobile && handleDragOver(e, index)}
              onDrop={() => !isMobile && handleDrop(index)}
              onDragEnd={() => { setDraggedIndex(null); setDragOverIndex(null); }}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: isMobile ? '10px' : '12px',
                padding: isMobile ? '12px 12px' : '14px 16px',
                marginBottom: '8px',
                background: dragOverIndex === index
                  ? 'rgba(74, 158, 255, 0.2)'
                  : draggedIndex === index
                    ? 'rgba(74, 158, 255, 0.1)'
                    : 'rgba(255, 255, 255, 0.03)',
                border: dragOverIndex === index
                  ? '1px solid rgba(74, 158, 255, 0.5)'
                  : '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '12px',
                cursor: isMobile ? 'pointer' : 'grab',
                transition: 'all 0.2s ease',
                transform: draggedIndex === index ? 'scale(1.02)' : 'scale(1)',
              }}
            >
              {/* Drag Handle - Hide on mobile */}
              {!isMobile && <GripVertical size={18} color="#555" style={{ cursor: 'grab' }} />}

              {/* Order Number */}
              <div style={{
                width: isMobile ? '24px' : '28px',
                height: isMobile ? '24px' : '28px',
                borderRadius: '50%',
                background: 'var(--accent)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: isMobile ? '11px' : '12px',
                fontWeight: '700',
                flexShrink: 0,
              }}>
                {index + 1}
              </div>

              {/* Song Info */}
              <div
                style={{ flex: 1, cursor: 'pointer' }}
                onClick={() => handleSongRequest(song.id)}
                title="Click to request this song near another"
              >
                <div style={{ fontWeight: '500', fontSize: '14px' }}>{song.name}</div>
                <div style={{ fontSize: '11px', color: '#888', marginTop: '2px' }}>
                  {song.bpm > 0 ? `${song.bpm} BPM` : 'BPM pending'} • {song.key || 'Key pending'}
                </div>
              </div>

              {/* Transition Arrow (except last) */}
              {index < playlist.length - 1 && (
                <ArrowRight size={16} color="var(--accent)" style={{ opacity: 0.5 }} />
              )}

              {/* Remove Button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRemoveSong(song.id);
                }}
                style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  background: hoveredIndex === index ? 'rgba(239, 68, 68, 0.2)' : 'transparent',
                  border: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  opacity: hoveredIndex === index ? 1 : 0.3,
                  transition: 'all 0.2s ease',
                }}
                title="Remove from playlist"
              >
                <X size={16} color="#ef4444" />
              </button>
            </div>
          ))}

          {playlist.length === 0 && (
            <div style={{
              textAlign: 'center',
              padding: isMobile ? '30px 20px' : '40px',
              color: '#666',
            }}>
              <Music size={isMobile ? 40 : 48} style={{ opacity: 0.3, marginBottom: '16px' }} />
              <p>No songs in playlist</p>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div style={{
          padding: isMobile ? '14px 12px' : '20px 28px',
          paddingBottom: isMobile ? 'max(14px, env(safe-area-inset-bottom))' : '20px',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: isMobile ? '10px' : '12px',
          justifyContent: 'flex-end',
        }}>
          <button
            onClick={onCancel}
            style={{
              padding: isMobile ? '14px 20px' : '12px 24px',
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: '10px',
              color: '#aaa',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              order: isMobile ? 2 : 1,
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(playlist)}
            disabled={playlist.length < 2}
            style={{
              padding: isMobile ? '14px 20px' : '12px 32px',
              background: playlist.length < 2
                ? 'rgba(100, 100, 100, 0.3)'
                : 'var(--accent)',
              border: 'none',
              borderRadius: '10px',
              color: playlist.length < 2 ? '#666' : '#fff',
              fontSize: '14px',
              fontWeight: '700',
              cursor: playlist.length < 2 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              boxShadow: playlist.length >= 2 ? '0 4px 20px rgba(74, 158, 255, 0.4)' : 'none',
              transition: 'all 0.2s ease',
              order: isMobile ? 1 : 2,
            }}
          >
            <CheckCircle size={18} />
            {isMobile ? 'Confirm & Mix' : 'Confirm & Start Mixing'}
          </button>
        </div>
      </div>

      {/* Song Request Modal */}
      {showSongRequestModal && selectedSongForRequest && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 2100,
        }}>
          <div style={{
            width: '400px',
            background: 'rgba(30, 30, 40, 0.98)',
            border: '1px solid rgba(74, 158, 255, 0.3)',
            borderRadius: '16px',
            padding: '24px',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
          }}>
            <h3 style={{ margin: '0 0 8px', fontSize: '16px', fontWeight: '600' }}>
              Song Request
            </h3>
            <p style={{ margin: '0 0 16px', fontSize: '13px', color: '#888' }}>
              Move &quot;{playlist.find(s => s.id === selectedSongForRequest)?.name}&quot; to play near:
            </p>
            <div style={{ maxHeight: '250px', overflow: 'auto' }}>
              {playlist
                .filter(s => s.id !== selectedSongForRequest)
                .map((song) => (
                  <button
                    key={song.id}
                    onClick={() => handleConfirmSongRequest(song.id)}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      marginBottom: '6px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: '#e0e0e0',
                      fontSize: '13px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(74, 158, 255, 0.1)';
                      e.currentTarget.style.borderColor = 'rgba(74, 158, 255, 0.3)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
                      e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                    }}
                  >
                    {song.name}
                  </button>
                ))}
            </div>
            <button
              onClick={() => {
                setShowSongRequestModal(false);
                setSelectedSongForRequest(null);
              }}
              style={{
                width: '100%',
                marginTop: '12px',
                padding: '10px',
                background: 'transparent',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                color: '#888',
                fontSize: '13px',
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
