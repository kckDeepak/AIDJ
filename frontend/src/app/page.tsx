/**
 * AI DJ Mixing System - with Enhanced Music Library
 */

'use client';

import { useState, useEffect, useRef } from 'react';
import { FuturisticBackground } from '@/components/FuturisticBackground';
import { AIBot } from '@/components/AIBot';
import { EnergyWave } from '@/components/EnergyWave';
import { EnergyWires } from '@/components/EnergyWires';
import { MusicLibrary } from '@/components/MusicLibrary';
import { AIDJLogo } from '@/components/AIDJLogo';
import { LibraryView } from '@/components/LibraryView';
import { ProfileView } from '@/components/ProfileView';
import { Navigation } from '@/components/Navigation';
import { GlobalMusicPlayer } from '@/components/GlobalMusicPlayer';
import { PlaylistConfirmation } from '@/components/PlaylistConfirmation';
import { SoundStrings, getBlendedGradient } from '@/components/SoundStrings';
import { Disclaimer } from '@/components/Disclaimer';

// API Configuration - uses environment variable in production
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_BASE_URL = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');

interface Song {
  id: string;
  name: string;
  bpm: number;
  key: string;
  file: File;
  isGenerated?: boolean;
  duration?: number;
  url?: string;  // URL for audio playback
}

interface PipelineStage {
  number: number;
  name: string;
  status: 'pending' | 'running' | 'complete';
}

// Workflow states for the new flow
type WorkflowState = 'input' | 'generating-playlist' | 'confirm-playlist' | 'mixing' | 'complete';

export default function HomePage() {
  const [songs, setSongs] = useState<Song[]>([]);
  const [selectedSongIds, setSelectedSongIds] = useState<Set<string>>(new Set());
  const [instructions, setInstructions] = useState('');
  const [librarySearchQuery, setLibrarySearchQuery] = useState('');
  const [isRemixing, setIsRemixing] = useState(false);
  const [hasRemixed, setHasRemixed] = useState(false);
  const [mixUrl, setMixUrl] = useState('');
  const [currentJobId, setCurrentJobId] = useState('');

  // New workflow state
  const [workflowState, setWorkflowState] = useState<WorkflowState>('input');
  const [generatedPlaylist, setGeneratedPlaylist] = useState<Song[]>([]);
  const [occasionPrompt, setOccasionPrompt] = useState('');
  const [isPaused, setIsPaused] = useState(false);

  // Remix naming
  const [remixTitle, setRemixTitle] = useState('');
  const [remixCounter, setRemixCounter] = useState(1);
  const [hasEditedTitle, setHasEditedTitle] = useState(false);

  const [stages, setStages] = useState<PipelineStage[]>([
    { number: 1, name: 'Song Selection', status: 'pending' },
    { number: 2, name: 'BPM Analysis', status: 'pending' },
    { number: 3, name: 'Structure Detection', status: 'pending' },
    { number: 4, name: 'Mix Planning', status: 'pending' },
    { number: 5, name: 'Mix Generation', status: 'pending' },
  ]);
  const [progressPercent, setProgressPercent] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);

  const [botState, setBotState] = useState<'idle' | 'analyzing' | 'remixing' | 'rendering' | 'error' | 'complete'>('idle');
  const [waveIntensity, setWaveIntensity] = useState(0);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [activeTab, setActiveTab] = useState<'remix' | 'library' | 'profile'>('remix');

  // Music Player State
  const [currentPlayingSong, setCurrentPlayingSong] = useState<Song | null>(null);
  const [playerCurrentTime, setPlayerCurrentTime] = useState(0);
  const [playerDuration, setPlayerDuration] = useState(180);
  const [playbackMode, setPlaybackMode] = useState<'order' | 'loop' | 'shuffle'>('order');
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Mobile responsive state
  const [isMobile, setIsMobile] = useState(false);
  const [showMobileLibrary, setShowMobileLibrary] = useState(false);

  // Check for mobile viewport
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);


  // Update wave intensity
  useEffect(() => {
    if (isRemixing) {
      setWaveIntensity(0.8);
    } else if (selectedSongIds.size > 0) {
      setWaveIntensity(0.4);
    } else if (instructions.length > 0) {
      setWaveIntensity(0.3);
    } else {
      setWaveIntensity(0.1);
    }
  }, [isRemixing, selectedSongIds.size, instructions.length]);

  // Update bot state
  useEffect(() => {
    if (!isRemixing && !hasRemixed) {
      setBotState('idle');
    } else if (hasRemixed) {
      setBotState('complete');
    } else {
      const currentStage = stages.find(s => s.status === 'running');
      if (currentStage) {
        if (currentStage.number === 1 || currentStage.number === 2) {
          setBotState('analyzing');
        } else if (currentStage.number === 3 || currentStage.number === 4) {
          setBotState('remixing');
        } else if (currentStage.number === 5) {
          setBotState('rendering');
        }
      }
    }
  }, [isRemixing, hasRemixed, stages]);

  // WebSocket connection
  useEffect(() => {
    if (!currentJobId) return;

    const wsUrl = `${WS_BASE_URL}/api/mix/ws/${currentJobId}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'stage_update') {
        setStages((prev) =>
          prev.map((s) =>
            s.number === message.stage ? { ...s, status: message.status } : s
          )
        );
      } else if (message.type === 'progress') {
        setProgressPercent(message.percent);
      } else if (message.type === 'log') {
        setLogs((prev) => [...prev, message.message].slice(-10));
      } else if (message.type === 'complete') {
        console.log('🎉 Mix generation complete!');
        console.log('Mix URL received:', message.mix_url);
        
        // Validate URL
        if (!message.mix_url) {
          console.error('❌ No mix URL received from backend!');
          alert('Mix generated but URL is missing. Check backend logs.');
          setIsRemixing(false);
          setBotState('error');
          return;
        }
        
        // Convert relative URL to absolute URL
        let fullMixUrl = message.mix_url;
        if (!message.mix_url.startsWith('http')) {
          fullMixUrl = `${API_BASE_URL}${message.mix_url}`;
        }
        console.log('Full mix URL:', fullMixUrl);
        
        setMixUrl(fullMixUrl);
        setIsRemixing(false);
        setHasRemixed(true);
        setWorkflowState('complete');

        // Use the custom title or auto-generated name
        const finalTitle = remixTitle.trim() || `Remix ${remixCounter}`;
        
        // Create unique ID for this remix
        const remixId = `remix_${Date.now()}`;
        
        console.log('Creating remix song with ID:', remixId);
        console.log('Remix URL:', fullMixUrl);

        const remixedSong: Song = {
          id: remixId,
          name: finalTitle,
          bpm: 120,
          key: 'C',
          file: new File([], remixId),
          isGenerated: true,
          url: fullMixUrl,
        };
        
        console.log('Remix song object:', remixedSong);
        setSongs((prev) => [...prev, remixedSong]);

        // Auto-play the remix in GlobalMusicPlayer
        setTimeout(() => {
          console.log('Auto-playing remix...');
          handlePlaySong(remixedSong);
        }, 500);
      } else if (message.type === 'error') {
        console.error('Mix generation failed:', message.message);
        setIsRemixing(false);
        setBotState('error');
        setWorkflowState('input');
        alert('Mix generation failed: ' + message.message);
      } else if (message.type === 'paused') {
        setIsPaused(true);
      } else if (message.type === 'resumed') {
        setIsPaused(false);
      } else if (message.type === 'cancelled') {
        setIsRemixing(false);
        setIsPaused(false);
        setWorkflowState('input');
        setCurrentJobId('');
        setStages((prev) => prev.map((s) => ({ ...s, status: 'pending' })));
        setProgressPercent(0);
        setLogs([]);
      }
    };

    return () => ws.close();
  }, [currentJobId]);


  // File upload - Upload to local storage via Backend API
  async function handleFileUpload(files: FileList | null) {
    if (!files || files.length === 0) {
      console.log('No files selected');
      return;
    }

    console.log(`Uploading ${files.length} file(s) to backend`);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];

      try {
        console.log(`[${i + 1}/${files.length}] Processing: ${file.name}`);

        // Validate file
        if (!file.name.toLowerCase().endsWith('.mp3')) {
          alert(`❌ ${file.name}: Only MP3 files are allowed`);
          continue;
        }
        
        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
          alert(`❌ ${file.name}: File too large (max 50MB)`);
          continue;
        }

        // Upload to backend
        console.log('📤 Uploading to backend...');
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/upload-audio`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Upload failed');
        }

        const uploadResult = await response.json();
        console.log('✅ Upload complete. URL:', uploadResult.url);

        // Add to local state
        const newSong: Song = {
          id: uploadResult.filename,
          name: uploadResult.filename.replace('.mp3', ''),
          bpm: 0,
          key: 'Unknown',
          file,
          url: `${API_BASE_URL}${uploadResult.url}`,
        };
        setSongs((prev) => [...prev, newSong]);

        console.log(`✅ Successfully uploaded: ${file.name}`);
      } catch (error) {
        console.error('❌ Upload error:', error);
        alert(`❌ Upload failed for ${file.name}: ${error}`);
      }
    }

    console.log('✅ All uploads complete');

    // Reload song list with metadata from backend
    try {
      console.log('🔄 Reloading songs with metadata...');
      const response = await fetch(`${API_BASE_URL}/api/songs`);
      
      if (!response.ok) {
        console.error('❌ Failed to fetch songs:', response.status);
        return;
      }

      const data = await response.json();
      
      if (data && data.songs && Array.isArray(data.songs)) {
        const loadedSongs: Song[] = data.songs.map((song: any) => ({
          id: song.filename,
          name: song.title,
          bpm: song.bpm || 0,
          key: song.key || 'Unknown',
          file: new File([], song.filename),
          url: `${API_BASE_URL}/static/songs/${song.filename}`,
          duration: song.duration,
        }));
        
        setSongs(loadedSongs);
        console.log('✅ Song list reloaded:', loadedSongs.length, 'songs');
      }
    } catch (error) {
      console.error('❌ Failed to reload songs:', error);
    }
  }

  // Delete song
  async function handleDelete(filename: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/songs/${filename}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setSongs((prev) => prev.filter(song => song.id !== filename));
        setSelectedSongIds((prev) => {
          const newSet = new Set(prev);
          newSet.delete(filename);
          return newSet;
        });
      }
    } catch (error) {
      console.error('Delete failed:', error);
    }
  }

  // Rename song - calls backend API
  async function handleRename(filename: string, newName: string) {
    // For generated songs, just update locally
    const song = songs.find(s => s.id === filename);
    if (song?.isGenerated) {
      setSongs((prev) =>
        prev.map((s) =>
          s.id === filename ? { ...s, name: newName } : s
        )
      );
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/songs/${encodeURIComponent(filename)}/rename?new_name=${encodeURIComponent(newName)}`,
        { method: 'PUT' }
      );

      if (response.ok) {
        const data = await response.json();
        // Update the song with new filename and name
        setSongs((prev) =>
          prev.map((s) =>
            s.id === filename ? { ...s, id: data.new_filename, name: data.title } : s
          )
        );
        // Update selected songs if this song was selected
        setSelectedSongIds((prev) => {
          if (prev.has(filename)) {
            const newSet = new Set(prev);
            newSet.delete(filename);
            newSet.add(data.new_filename);
            return newSet;
          }
          return prev;
        });
      } else {
        const error = await response.json();
        alert(`Rename failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Rename failed:', error);
      alert('Rename failed. Please try again.');
    }
  }

  // Reorder songs
  function handleReorder(fromIndex: number, toIndex: number) {
    setSongs(prevSongs => {
      const newSongs = [...prevSongs];
      const [removed] = newSongs.splice(fromIndex, 1);
      newSongs.splice(toIndex, 0, removed);
      return newSongs;
    });
  }

  // Music Player Handlers
  function handlePlaySong(song: Song) {
    console.log('🎵 handlePlaySong called with:', {
      id: song.id,
      name: song.name,
      isGenerated: song.isGenerated,
      url: song.url,
    });
    
    // If same song, just toggle play/pause
    if (currentPlayingSong?.id === song.id) {
      handlePlayPause();
      return;
    }

    setCurrentPlayingSong(song);
    setPlayerCurrentTime(0);

    // Validate that we have a URL
    if (!song.url) {
      console.error('❌ No URL available for song:', song.name);
      alert('Failed to play: Song URL not available');
      return;
    }

    const audioUrl = song.url;
    console.log('🔊 Playing audio from:', audioUrl);

    // Create or update audio element
    if (audioRef.current) {
      audioRef.current.pause();
    }

    const audio = new Audio(audioUrl);
    audioRef.current = audio;

    audio.addEventListener('loadedmetadata', () => {
      console.log('✅ Audio metadata loaded. Duration:', audio.duration);
      setPlayerDuration(audio.duration);
    });

    audio.addEventListener('timeupdate', () => {
      setPlayerCurrentTime(audio.currentTime);
    });

    audio.addEventListener('ended', () => {
      handleNextSong();
    });

    audio.addEventListener('error', (e) => {
      console.error('❌ Audio playback error:', e);
      console.error('❌ Failed URL:', audioUrl);
      console.error('❌ Error details:', {
        code: (e.target as HTMLAudioElement)?.error?.code,
        message: (e.target as HTMLAudioElement)?.error?.message,
      });
      setIsAudioPlaying(false);
      alert(`Failed to play audio. URL: ${audioUrl}\nError: ${(e.target as HTMLAudioElement)?.error?.message || 'Unknown error'}`);
    });

    audio.play().then(() => {
      console.log('✅ Audio playback started');
      setIsAudioPlaying(true);
    }).catch(err => {
      console.error('❌ Playback failed:', err);
      console.error('❌ Failed URL:', audioUrl);
      setIsAudioPlaying(false);
      alert(`Playback failed: ${err.message}`);
    });
  }

  function handlePlayPause() {
    if (!audioRef.current) return;

    if (isAudioPlaying) {
      audioRef.current.pause();
      setIsAudioPlaying(false);
    } else {
      audioRef.current.play().then(() => {
        setIsAudioPlaying(true);
      }).catch(err => {
        console.error('Playback failed:', err);
      });
    }
  }

  function handlePreviousSong() {
    if (!currentPlayingSong) return;
    const currentIndex = songs.findIndex(s => s.id === currentPlayingSong.id);
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : songs.length - 1;
    if (songs[prevIndex]) {
      handlePlaySong(songs[prevIndex]);
    }
  }

  function handleNextSong() {
    if (!currentPlayingSong || songs.length === 0) return;
    const currentIndex = songs.findIndex(s => s.id === currentPlayingSong.id);

    if (playbackMode === 'shuffle') {
      const randomIndex = Math.floor(Math.random() * songs.length);
      handlePlaySong(songs[randomIndex]);
    } else if (playbackMode === 'loop') {
      // Restart current song
      if (audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play();
      }
    } else {
      const nextIndex = currentIndex < songs.length - 1 ? currentIndex + 1 : 0;
      handlePlaySong(songs[nextIndex]);
    }
  }

  function handleSeek(time: number) {
    setPlayerCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  }

  function handleClosePlayer() {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setCurrentPlayingSong(null);
    setIsAudioPlaying(false);
    setPlayerCurrentTime(0);
  }

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);


  // Download song
  async function handleDownload(filename: string) {
    try {
      // Check if it's a generated song
      const song = songs.find(s => s.id === filename);
      
      // Require URL
      if (!song?.url) {
        alert('Download failed: Song URL not available');
        return;
      }
      
      const url = song.url;

      const response = await fetch(url);
      if (response.ok) {
        const blob = await response.blob();
        const blobUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        // Use song name for download filename
        const downloadName = song?.name ? `${song.name}.mp3` : filename;
        a.download = downloadName;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(blobUrl);
        document.body.removeChild(a);
      } else {
        alert('Download failed. File may not exist.');
      }
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    }
  }

  // Load songs from backend
  useEffect(() => {
    async function loadSongs() {
      try {
        // Fetch songs with metadata from backend
        const response = await fetch(`${API_BASE_URL}/api/songs`);

        if (!response.ok) {
          console.error('❌ Failed to fetch songs:', response.status);
          setSongs([]);
          return;
        }

        const data = await response.json();

        // Safety check - ensure data.songs exists and is an array
        if (!data || !data.songs || !Array.isArray(data.songs)) {
          console.error('❌ Invalid songs response:', data);
          setSongs([]);
          return;
        }

        const loadedSongs: Song[] = data.songs.map((song: any) => ({
          id: song.filename,
          name: song.title,
          bpm: song.bpm || 0,
          key: song.key || 'Unknown',
          file: new File([], song.filename),
          url: `${API_BASE_URL}/static/songs/${song.filename}`,
          duration: song.duration,
        }));

        setSongs(loadedSongs);
        console.log(`✅ Loaded ${loadedSongs.length} songs with metadata`);
      } catch (error) {
        console.error('❌ Failed to load songs:', error);
        setSongs([]); // Set empty array on error
      }
    }

    loadSongs();
  }, []);

  function toggleSongSelection(songId: string) {
    setSelectedSongIds((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(songId)) {
        newSet.delete(songId);
      } else {
        newSet.add(songId);
      }
      return newSet;
    });
  }

  async function handleRemix() {
    const selectedSongs = songs.filter((song) => selectedSongIds.has(song.id));

    let prompt = instructions || 'Mix the following songs: ';
    prompt += selectedSongs.map(s => s.name).join(', ');

    setIsRemixing(true);
    setProgressPercent(0);
    setLogs([]);
    setStages((prev) => prev.map((s) => ({ ...s, status: 'pending' })));
    setWorkflowState('mixing');

    try {
      const response = await fetch(`${API_BASE_URL}/api/mix/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentJobId(data.job_id);
      } else {
        throw new Error('Failed to start mix generation');
      }
    } catch (error) {
      console.error('Remix failed:', error);
      setIsRemixing(false);
      setBotState('error');
      setWorkflowState('input');
      alert('Failed to start remix: ' + error);
    }
  }

  // Generate playlist from natural language prompt
  async function handleGeneratePlaylist() {
    if (!occasionPrompt.trim()) {
      alert('Please describe the occasion or vibe for your mix');
      return;
    }

    setWorkflowState('generating-playlist');
    setBotState('analyzing');

    // For now, we'll use the songs from selection or all songs
    // In a full implementation, this would call an AI endpoint
    const selectedSongs = songs.filter((song) => selectedSongIds.has(song.id));
    const playlistSongs = selectedSongs.length > 0 ? selectedSongs : songs.slice(0, 5);

    // Simulate AI processing
    setTimeout(() => {
      setGeneratedPlaylist(playlistSongs);
      setWorkflowState('confirm-playlist');
      setBotState('idle');
    }, 1500);
  }

  // Confirm playlist and start mixing
  async function handleConfirmPlaylist(confirmedPlaylist: Song[]) {
    setGeneratedPlaylist(confirmedPlaylist);
    setWorkflowState('mixing');

    // Update selected song IDs to match confirmed playlist
    const newSelectedIds = new Set(confirmedPlaylist.map(s => s.id));
    setSelectedSongIds(newSelectedIds);

    // Build prompt from occasion and songs
    let prompt = occasionPrompt + '. Mix these songs in order: ';
    prompt += confirmedPlaylist.map(s => s.name).join(', ');

    setIsRemixing(true);
    setProgressPercent(0);
    setLogs([]);
    setStages((prev) => prev.map((s) => ({ ...s, status: 'pending' })));

    try {
      const response = await fetch(`${API_BASE_URL}/api/mix/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentJobId(data.job_id);
      } else {
        throw new Error('Failed to start mix generation');
      }
    } catch (error) {
      console.error('Mix failed:', error);
      setIsRemixing(false);
      setBotState('error');
      setWorkflowState('input');
      alert('Failed to start mix: ' + error);
    }
  }

  // Cancel playlist confirmation
  function handleCancelPlaylist() {
    setWorkflowState('input');
    setGeneratedPlaylist([]);
  }

  // Handle song request (move song near another)
  function handleSongRequest(songId: string, nearSongId: string) {
    setGeneratedPlaylist(prev => {
      const songIndex = prev.findIndex(s => s.id === songId);
      const nearIndex = prev.findIndex(s => s.id === nearSongId);

      if (songIndex === -1 || nearIndex === -1) return prev;

      const newPlaylist = [...prev];
      const [song] = newPlaylist.splice(songIndex, 1);

      // Insert after the target song
      const insertIndex = songIndex < nearIndex ? nearIndex : nearIndex + 1;
      newPlaylist.splice(insertIndex, 0, song);

      return newPlaylist;
    });
  }

  // Remove song from generated playlist
  function handleRemoveFromPlaylist(songId: string) {
    setGeneratedPlaylist(prev => prev.filter(s => s.id !== songId));
  }

  // Reorder playlist
  function handleReorderPlaylist(fromIndex: number, toIndex: number) {
    setGeneratedPlaylist(prev => {
      const newPlaylist = [...prev];
      const [removed] = newPlaylist.splice(fromIndex, 1);
      newPlaylist.splice(toIndex, 0, removed);
      return newPlaylist;
    });
  }

  // Pause mixing
  async function handlePauseMix() {
    if (!currentJobId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/mix/pause/${currentJobId}`, {
        method: 'POST',
      });

      if (response.ok) {
        setIsPaused(true);
      }
    } catch (error) {
      console.error('Failed to pause:', error);
    }
  }

  // Resume mixing
  async function handleResumeMix() {
    if (!currentJobId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/mix/resume/${currentJobId}`, {
        method: 'POST',
      });

      if (response.ok) {
        setIsPaused(false);
      }
    } catch (error) {
      console.error('Failed to resume:', error);
    }
  }

  // Cancel mixing
  async function handleCancelMix() {
    if (!currentJobId) return;

    if (!confirm('Are you sure you want to cancel the mix generation?')) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/mix/cancel/${currentJobId}`, {
        method: 'POST',
      });

      if (response.ok) {
        setIsRemixing(false);
        setIsPaused(false);
        setWorkflowState('input');
        setCurrentJobId('');
      }
    } catch (error) {
      console.error('Failed to cancel:', error);
    }
  }

  const selectedSongs = songs.filter((song) => selectedSongIds.has(song.id));
  const wireStage = botState === 'analyzing' ? 'analyzing' :
    botState === 'remixing' ? 'mixing' :
      botState === 'rendering' ? 'rendering' :
        'idle';

  return (
    <div style={{
      width: '100vw',
      minHeight: '100vh',
      height: currentPlayingSong ? 'calc(100vh + 80px)' : '100vh',
      margin: 0,
      padding: 0,
      paddingBottom: currentPlayingSong ? (isMobile ? '100px' : '80px') : '0',
      backgroundColor: '#0a0a0a',
      color: '#e0e0e0',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      position: 'relative',
      overflow: 'hidden',
      transition: 'height 0.3s ease, padding-bottom 0.3s ease',
    }}>
      <FuturisticBackground />

      {/* HEADER ROW - Logo and Future Controls */}
      <div style={{
        position: 'relative',
        zIndex: 100,
        height: isMobile ? '60px' : '70px',
        display: 'flex',
        alignItems: 'center',
        padding: isMobile ? '0 12px' : '0 20px',
        background: 'rgba(10, 10, 15, 0.6)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        gap: isMobile ? '8px' : '0',
      }}>
        {/* Mobile Library Toggle */}
        {isMobile && activeTab === 'remix' && (
          <button
            onClick={() => setShowMobileLibrary(!showMobileLibrary)}
            style={{
              padding: '10px',
              background: showMobileLibrary ? 'var(--accent-bg)' : 'rgba(255, 255, 255, 0.05)',
              border: showMobileLibrary ? '1px solid var(--accent-border)' : '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '10px',
              color: showMobileLibrary ? 'var(--accent)' : 'var(--gray-300)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease',
            }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 18V5l12-2v13" />
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="16" r="3" />
            </svg>
          </button>
        )}

        {/* Logo Container */}
        <div style={{ position: 'static', transform: isMobile ? 'scale(0.85)' : 'none', transformOrigin: 'left center' }}>
          <AIDJLogo onRemixComplete={hasRemixed} />
        </div>

        {/* Navigation Tabs */}
        <Navigation activeTab={activeTab} onTabChange={setActiveTab} isMobile={isMobile} />
      </div>

      {/* Remix plays in GlobalMusicPlayer now */}

      {/* Main Content */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        padding: isMobile ? '12px' : '20px',
        gap: isMobile ? '12px' : '20px',
        overflow: 'auto',
      }}>
        {/* View Switching */}
        {activeTab === 'remix' && (
          <>
            <div style={{
              display: 'flex',
              flexDirection: isMobile ? 'column' : 'row',
              flex: 1,
              gap: isMobile ? '12px' : '20px',
              minHeight: isMobile ? 'auto' : '500px',
              paddingBottom: currentPlayingSong ? (isMobile ? '100px' : '80px') : '0',
              transition: 'padding-bottom 0.3s ease',
              position: 'relative',
            }}>
              {/* Mobile Library Overlay */}
              {isMobile && showMobileLibrary && (
                <div
                  style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: 'rgba(0, 0, 0, 0.6)',
                    zIndex: 200,
                  }}
                  onClick={() => setShowMobileLibrary(false)}
                />
              )}

              {/* LEFT PANEL - Enhanced Music Library */}
              <div style={{
                width: isMobile ? (showMobileLibrary ? '85%' : '0') : 'clamp(280px, 28%, 400px)',
                maxWidth: isMobile ? '320px' : 'none',
                display: 'flex',
                flexDirection: 'column',
                padding: isMobile && !showMobileLibrary ? '0' : '20px',
                background: 'rgba(20, 20, 30, 0.4)',
                backdropFilter: 'blur(16px) saturate(180%)',
                WebkitBackdropFilter: 'blur(16px) saturate(180%)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '20px',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1)',
                overflow: isMobile && !showMobileLibrary ? 'hidden' : 'visible',
                transition: 'all 0.3s ease',
                opacity: isRemixing ? 0.7 : 1,
                position: isMobile ? 'fixed' : 'relative',
                left: isMobile ? (showMobileLibrary ? '0' : '-100%') : 'auto',
                top: isMobile ? '60px' : 'auto',
                bottom: isMobile ? '0' : 'auto',
                zIndex: isMobile ? 201 : 'auto',
                height: isMobile ? 'calc(100vh - 60px)' : 'auto',
              }}>
                {/* Header with Search */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                  <h2 style={{ margin: 0, fontSize: isMobile ? '16px' : '18px', fontWeight: '600', whiteSpace: 'nowrap', color: 'var(--accent)' }}>Stack</h2>
                  {isMobile && (
                    <button
                      onClick={() => setShowMobileLibrary(false)}
                      style={{
                        marginLeft: 'auto',
                        padding: '8px',
                        background: 'transparent',
                        border: 'none',
                        color: '#888',
                        cursor: 'pointer',
                        fontSize: '20px',
                      }}
                    >
                      ✕
                    </button>
                  )}
                </div>
                <div style={{
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '10px',
                }}>
                  <input
                    type="text"
                    placeholder="Search..."
                    value={librarySearchQuery}
                    onChange={(e) => setLibrarySearchQuery(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '6px 12px 6px 32px',
                      background: 'rgba(255, 255, 255, 0.08)',
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      borderRadius: '8px',
                      color: '#e0e0e0',
                      fontSize: '13px',
                      outline: 'none',
                      transition: 'all 0.2s ease',
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.border = '1px solid var(--accent-border)';
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.12)';
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.border = '1px solid rgba(255, 255, 255, 0.15)';
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                    }}
                  />
                  <svg
                    style={{
                      position: 'absolute',
                      left: '10px',
                      width: '14px',
                      height: '14px',
                      color: '#888',
                      pointerEvents: 'none',
                    }}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>

                <AIBot state={botState} isPlaying={isAudioPlaying} progressPercent={progressPercent} />

                <MusicLibrary
                  songs={songs.filter(song =>
                    librarySearchQuery === '' ||
                    song.name.toLowerCase().includes(librarySearchQuery.toLowerCase()) ||
                    song.key.toLowerCase().includes(librarySearchQuery.toLowerCase()) ||
                    song.bpm.toString().includes(librarySearchQuery)
                  )}
                  selectedSongIds={selectedSongIds}
                  onToggleSelection={toggleSongSelection}
                  onFileUpload={handleFileUpload}
                  onDelete={handleDelete}
                  onRename={handleRename}
                  onDownload={handleDownload}
                  onReorder={handleReorder}
                  onPlay={(song) => handlePlaySong(song)}
                />
              </div>

              {/* MAIN PANEL - Glassmorphism */}
              <div style={{
                flex: 1,
                padding: isMobile ? '16px' : '20px',
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                background: 'rgba(20, 20, 30, 0.4)',
                backdropFilter: 'blur(16px) saturate(180%)',
                WebkitBackdropFilter: 'blur(16px) saturate(180%)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: isMobile ? '16px' : '20px',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1)',
                overflow: 'auto',
                minWidth: 0,
                minHeight: isMobile ? '0' : 'auto',
              }}>
                {/* EnergyWave removed for cleaner look */}

                <h2 style={{ margin: '0 0 12px 0', fontSize: isMobile ? '16px' : '18px', position: 'relative', zIndex: 2, fontWeight: '600', color: 'var(--accent)' }}>
                  Studio
                </h2>

                {/* Natural Language Input Section */}
                <div style={{ marginBottom: isMobile ? '15px' : '20px', position: 'relative', zIndex: 2 }}>
                  <label style={{ display: 'block', marginBottom: '6px', fontSize: isMobile ? '13px' : '14px', color: '#ccc', fontWeight: '500' }}>
                    Describe your vibe:
                  </label>
                  <textarea
                    value={occasionPrompt}
                    onChange={(e) => setOccasionPrompt(e.target.value)}
                    placeholder={isMobile ? "e.g., 'Beach party with tropical vibes'" : "e.g., 'Create me a mix for a beach party with tropical vibes' or 'Chill Sunday morning brunch mix with smooth transitions'"}
                    disabled={isRemixing}
                    style={{
                      width: '100%',
                      minHeight: isMobile ? '60px' : '80px',
                      padding: isMobile ? '12px' : '14px 16px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: occasionPrompt.length > 0 ? '1px solid var(--accent-border)' : '1px solid rgba(255, 255, 255, 0.15)',
                      borderRadius: '12px',
                      color: '#e0e0e0',
                      fontSize: isMobile ? '14px' : '14px',
                      resize: 'vertical',
                      transition: 'all 0.3s ease',
                      boxShadow: occasionPrompt.length > 0 ? '0 0 20px var(--accent-glow)' : 'none',
                      fontFamily: 'inherit',
                    }}
                  />
                </div>

                {/* Remix Title Input */}
                {selectedSongs.length > 0 && (
                  <div style={{ marginBottom: '15px', position: 'relative', zIndex: 2 }}>
                    <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#aaa', fontWeight: '500' }}>
                      Remix Name:
                    </label>
                    <input
                      type="text"
                      value={hasEditedTitle ? remixTitle : `Remix ${remixCounter}`}
                      onChange={(e) => {
                        setRemixTitle(e.target.value);
                        setHasEditedTitle(true);
                      }}
                      onFocus={() => {
                        if (!hasEditedTitle) {
                          setRemixTitle(`Remix ${remixCounter}`);
                        }
                      }}
                      placeholder={`Remix ${remixCounter}`}
                      disabled={isRemixing}
                      style={{
                        width: '100%',
                        padding: '12px 16px',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(168, 85, 247, 0.3)',
                        borderRadius: '10px',
                        color: '#e0e0e0',
                        fontSize: '16px',
                        fontWeight: '600',
                        transition: 'all 0.3s ease',
                        boxShadow: '0 0 15px rgba(168, 85, 247, 0.15)',
                      }}
                    />
                  </div>
                )}

                {/* Selected Songs Display with Sound Strings */}
                <div style={{ marginBottom: '15px', position: 'relative', zIndex: 2 }}>
                  <h3 style={{ fontSize: '13px', marginBottom: '10px', color: '#aaa', fontWeight: '500' }}>
                    Selected Songs ({selectedSongs.length}):
                  </h3>
                  <div style={{
                    position: 'relative',
                    minHeight: selectedSongs.length > 0 ? '160px' : '60px',
                    padding: '10px',
                    background: selectedSongs.length > 0 ? 'rgba(0, 0, 0, 0.2)' : 'transparent',
                    borderRadius: '16px',
                    overflow: 'hidden',
                  }}>
                    {/* Animated Sound Strings Background */}
                    {selectedSongs.length > 0 && (
                      <SoundStrings
                        songCount={selectedSongs.length}
                        isRemixing={isRemixing}
                        progressPercent={progressPercent}
                        stage={isRemixing ? (botState === 'analyzing' ? 'analyzing' : botState === 'rendering' ? 'rendering' : 'mixing') : 'idle'}
                      />
                    )}

                    {/* Song Cards Grid */}
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
                      gap: '10px',
                      position: 'relative',
                      zIndex: 1,
                    }}>
                      {selectedSongs.map((song, idx) => (
                        <div
                          key={song.id}
                          style={{
                            aspectRatio: '1',
                            background: 'rgba(20, 20, 30, 0.8)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            borderRadius: '12px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: '10px',
                            textAlign: 'center',
                            position: 'relative',
                            zIndex: 3,
                            boxShadow: isRemixing ? `0 0 20px var(--accent-glow)` : 'none',
                            transition: 'all 0.5s ease',
                            animation: isRemixing ? `pulse 2s ease-in-out infinite ${idx * 0.2}s` : 'none',
                            fontSize: '12px',
                            fontWeight: '500',
                          }}
                        >
                          {song.name}
                        </div>
                      ))}
                      {selectedSongs.length === 0 && (
                        <div style={{
                          color: 'var(--accent)',
                          position: 'relative',
                          zIndex: 3,
                          fontSize: '13px',
                          padding: '20px',
                          animation: 'purplePulse 2s ease-in-out infinite',
                        }}>
                          Select your stack to vibe
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: isMobile ? '10px' : '12px', marginBottom: '15px', position: 'relative', zIndex: 2 }}>
                  {/* Generate Playlist Button */}
                  <button
                    onClick={handleGeneratePlaylist}
                    disabled={!occasionPrompt.trim() || isRemixing}
                    style={{
                      flex: 1,
                      padding: isMobile ? '14px 20px' : '14px 28px',
                      background: !occasionPrompt.trim() || isRemixing
                        ? 'rgba(100, 100, 100, 0.3)'
                        : 'var(--accent)',
                      color: !occasionPrompt.trim() || isRemixing ? '#666' : '#fff',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      borderRadius: '12px',
                      cursor: !occasionPrompt.trim() || isRemixing ? 'not-allowed' : 'pointer',
                      fontSize: isMobile ? '13px' : '14px',
                      fontWeight: '700',
                      transition: 'all 0.3s ease',
                      boxShadow: occasionPrompt.trim() && !isRemixing
                        ? '0 4px 20px var(--accent-glow)'
                        : 'none',
                      textTransform: 'uppercase',
                      letterSpacing: isMobile ? '0.5px' : '1px',
                    }}
                  >
                    {workflowState === 'generating-playlist' ? '✨ GENERATING...' : (isMobile ? '✨ GENERATE' : '✨ GENERATE PLAYLIST')}
                  </button>

                  {/* Quick Remix Button (if songs selected) */}
                  {selectedSongs.length > 0 && (
                    <button
                      onClick={handleRemix}
                      disabled={isRemixing}
                      style={{
                        padding: isMobile ? '14px 20px' : '14px 28px',
                        background: isRemixing
                          ? 'rgba(100, 100, 100, 0.3)'
                          : 'rgba(74, 222, 128, 0.2)',
                        color: isRemixing ? '#666' : '#4ade80',
                        border: '1px solid rgba(74, 222, 128, 0.4)',
                        borderRadius: '12px',
                        cursor: isRemixing ? 'not-allowed' : 'pointer',
                        fontSize: '14px',
                        fontWeight: '600',
                        transition: 'all 0.3s ease',
                      }}
                    >
                      QUICK MIX
                    </button>
                  )}
                </div>

                {/* Pipeline Progress with Pause/Cancel */}
                {isRemixing && (
                  <div style={{ marginTop: '15px', position: 'relative', zIndex: 2 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                      <h3 style={{ fontSize: '15px', fontWeight: '600', margin: 0 }}>Pipeline Progress</h3>

                      {/* Pause/Cancel Controls */}
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          onClick={isPaused ? handleResumeMix : handlePauseMix}
                          style={{
                            padding: '8px 16px',
                            background: isPaused ? 'rgba(74, 222, 128, 0.2)' : 'rgba(251, 191, 36, 0.2)',
                            color: isPaused ? '#4ade80' : '#fbbf24',
                            border: `1px solid ${isPaused ? 'rgba(74, 222, 128, 0.4)' : 'rgba(251, 191, 36, 0.4)'}`,
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontSize: '12px',
                            fontWeight: '600',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            transition: 'all 0.2s ease',
                          }}
                        >
                          {isPaused ? '▶ RESUME' : '⏸ PAUSE'}
                        </button>
                        <button
                          onClick={handleCancelMix}
                          style={{
                            padding: '8px 16px',
                            background: 'rgba(239, 68, 68, 0.2)',
                            color: '#ef4444',
                            border: '1px solid rgba(239, 68, 68, 0.4)',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontSize: '12px',
                            fontWeight: '600',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            transition: 'all 0.2s ease',
                          }}
                        >
                          ✕ CANCEL
                        </button>
                      </div>
                    </div>

                    {/* Paused indicator */}
                    {isPaused && (
                      <div style={{
                        padding: '10px 16px',
                        marginBottom: '12px',
                        background: 'rgba(251, 191, 36, 0.1)',
                        border: '1px solid rgba(251, 191, 36, 0.3)',
                        borderRadius: '8px',
                        color: '#fbbf24',
                        fontSize: '13px',
                        fontWeight: '500',
                        textAlign: 'center',
                      }}>
                        ⏸ Mix generation is paused
                      </div>
                    )}

                    <div style={{ marginBottom: '15px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                        <div style={{ fontSize: '11px', color: '#aaa' }}>
                          {Math.round(progressPercent)}%
                        </div>
                        <div style={{ fontSize: '11px', color: '#888' }}>
                          {stages.find(s => s.status === 'running')?.name || 'Starting...'}
                        </div>
                      </div>
                      <div style={{
                        height: '8px',
                        background: 'rgba(255, 255, 255, 0.1)',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        position: 'relative',
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${progressPercent}%`,
                          background: isPaused
                            ? 'linear-gradient(90deg, #fbbf24, #f59e0b)'
                            : getBlendedGradient(selectedSongs.length),
                          transition: 'width 0.5s ease',
                          boxShadow: isPaused
                            ? '0 0 15px rgba(251, 191, 36, 0.6)'
                            : '0 0 15px var(--accent-glow)',
                          animation: !isPaused ? 'shimmer 2s ease-in-out infinite' : 'none',
                        }} />
                      </div>
                    </div>

                    <div style={{ marginBottom: '15px' }}>
                      {stages.map((stage) => (
                        <div
                          key={stage.number}
                          style={{
                            padding: '8px 12px',
                            marginBottom: '4px',
                            background: stage.status === 'complete'
                              ? 'rgba(74, 222, 128, 0.15)'
                              : stage.status === 'running'
                                ? isPaused ? 'rgba(251, 191, 36, 0.15)' : 'var(--accent-bg)'
                                : 'rgba(255, 255, 255, 0.03)',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            borderRadius: '8px',
                            fontSize: '12px',
                            transition: 'all 0.5s ease',
                            boxShadow: stage.status === 'running'
                              ? isPaused ? '0 0 15px rgba(251, 191, 36, 0.3)' : '0 0 15px var(--accent-glow)'
                              : 'none',
                          }}
                        >
                          {stage.number}. {stage.name} - {stage.status === 'running' && isPaused ? 'paused' : stage.status}
                        </div>
                      ))}
                    </div>

                    {logs.length > 0 && (
                      <div style={{
                        background: 'rgba(0, 0, 0, 0.3)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        padding: '10px',
                        fontSize: '10px',
                        fontFamily: 'monospace',
                        maxHeight: '120px',
                        overflow: 'auto',
                        borderRadius: '8px',
                      }}>
                        {logs.map((log, idx) => (
                          <div key={idx} style={{ marginBottom: '2px', color: '#888' }}>
                            {log}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Disclaimer for Remix Tab - Separate at bottom */}
            <div style={{ padding: isMobile ? '0 12px 12px' : '0 20px 20px' }}>
              <Disclaimer />
            </div>
          </>
        )}

        {/* Library View */}
        {activeTab === 'library' && (
          <div style={{
            flex: 1,
            padding: isMobile ? '12px' : '20px',
            paddingBottom: currentPlayingSong ? (isMobile ? '110px' : '100px') : (isMobile ? '12px' : '20px'),
            background: 'rgba(20, 20, 30, 0.4)',
            backdropFilter: 'blur(16px) saturate(180%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: isMobile ? '16px' : '20px',
            overflow: 'auto',
            transition: 'padding-bottom 0.3s ease',
          }}>
            <LibraryView
              songs={songs}
              onDelete={handleDelete}
              onRename={handleRename}
              onDownload={handleDownload}
              onReorder={handleReorder}
              onPlay={(song) => handlePlaySong(song)}
            />
            <Disclaimer />
          </div>
        )}

        {/* Profile View */}
        {activeTab === 'profile' && (
          <div style={{
            flex: 1,
            padding: isMobile ? '12px' : '20px',
            paddingBottom: currentPlayingSong ? (isMobile ? '110px' : '100px') : (isMobile ? '12px' : '20px'),
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            background: 'transparent',
            border: 'none',
            borderRadius: isMobile ? '16px' : '20px',
            overflow: 'auto',
            transition: 'padding-bottom 0.3s ease',
          }}>
            <ProfileView
              totalSongs={songs.filter(s => !s.isGenerated).length}
              totalRemixes={songs.filter(s => s.isGenerated).length}
            />
            <div style={{ width: '100%', maxWidth: '500px', padding: '0 24px' }}>
              <Disclaimer />
            </div>
          </div>
        )}
      </div>

      <style jsx global>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }
        @keyframes slideUp {
          from {
            transform: translateX(-50%) translateY(100px);
            opacity: 0;
          }
          to {
            transform: translateX(-50%) translateY(0);
            opacity: 1;
          }
        }
        @keyframes shimmer {
          0% { filter: brightness(1); }
          50% { filter: brightness(1.3); }
          100% { filter: brightness(1); }
        }
        @keyframes purplePulse {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; text-shadow: 0 0 10px var(--accent-glow); }
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        ::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb {
          background: rgba(74, 158, 255, 0.3);
          border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(74, 158, 255, 0.5);
        }
      `}</style>

      {/* Global Music Player */}
      <GlobalMusicPlayer
        songs={songs}
        currentSong={currentPlayingSong}
        isPlaying={isAudioPlaying}
        onPlayPause={handlePlayPause}
        onPrevious={handlePreviousSong}
        onNext={handleNextSong}
        onSeek={handleSeek}
        onModeChange={setPlaybackMode}
        onClose={handleClosePlayer}
        currentTime={playerCurrentTime}
        duration={playerDuration}
        playbackMode={playbackMode}
      />

      {/* Playlist Confirmation Modal */}
      {workflowState === 'confirm-playlist' && (
        <PlaylistConfirmation
          playlist={generatedPlaylist}
          occasion={occasionPrompt}
          onConfirm={(playlist) => {
            handleConfirmPlaylist(playlist);
            // Increment remix counter and reset title for next remix
            if (!hasEditedTitle) {
              setRemixCounter(prev => prev + 1);
            }
            setHasEditedTitle(false);
            setRemixTitle('');
          }}
          onCancel={handleCancelPlaylist}
          onSongRequest={handleSongRequest}
          onRemoveSong={handleRemoveFromPlaylist}
          onReorderPlaylist={handleReorderPlaylist}
        />
      )}
    </div>
  );
}
