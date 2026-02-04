/**
 * API Client for AI DJ Mixing System Backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Song {
    filename: string;
    title: string;
    artist: string;
    bpm?: number;
    key?: string;
    genre?: string;
    energy?: number;
    duration?: number;
    url?: string;
}

export interface MixJob {
    job_id: string;
    status: string;
    message: string;
    websocket_url: string;
}

export interface MixStatus {
    job_id: string;
    status: string;
    current_stage: number;
    current_stage_name: string;
    progress_percent: number;
    logs: Array<{ message: string; level: string; time: string }>;
    error?: string;
    mix_url?: string;
}

export interface PipelineStage {
    number: number;
    name: string;
    description: string;
}

/**
 * Fetch all songs
 */
export async function fetchSongs(): Promise<{ songs: Song[]; total: number }> {
    const res = await fetch(`${API_BASE_URL}/api/songs`);
    if (!res.ok) throw new Error('Failed to fetch songs');
    return res.json();
}

/**
 * Upload a new song
 */
export async function uploadSong(file: File): Promise<Song> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE_URL}/api/songs/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) throw new Error('Failed to upload song');
    const data = await res.json();
    return data.metadata;
}

/**
 * Delete a song
 */
export async function deleteSong(filename: string): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/api/songs/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
    });

    if (!res.ok) throw new Error('Failed to delete song');
}

/**
 * Get waveform data for a song
 */
export async function fetchWaveform(filename: string, points = 100): Promise<number[]> {
    const res = await fetch(
        `${API_BASE_URL}/api/songs/${encodeURIComponent(filename)}/waveform?points=${points}`
    );

    if (!res.ok) throw new Error('Failed to fetch waveform');
    const data = await res.json();
    return data.waveform;
}

/**
 * Start mix generation
 */
export async function generateMix(prompt: string): Promise<MixJob> {
    const res = await fetch(`${API_BASE_URL}/api/mix/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
    });

    if (!res.ok) throw new Error('Failed to start mix generation');
    return res.json();
}

/**
 * Get mix status
 */
export async function fetchMixStatus(jobId: string): Promise<MixStatus> {
    const res = await fetch(`${API_BASE_URL}/api/mix/status/${jobId}`);
    if (!res.ok) throw new Error('Failed to fetch mix status');
    return res.json();
}

/**
 * Get pipeline stages
 */
export async function fetchPipelineStages(): Promise<{ stages: PipelineStage[] }> {
    const res = await fetch(`${API_BASE_URL}/api/mix/stages`);
    if (!res.ok) throw new Error('Failed to fetch pipeline stages');
    return res.json();
}

/**
 * Get WebSocket URL for real-time updates
 */
export function getWebSocketUrl(jobId: string): string {
    const wsBase = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');
    return `${wsBase}/api/mix/ws/${jobId}`;
}
