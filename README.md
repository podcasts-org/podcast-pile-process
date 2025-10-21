# Podcast Pile

Data processing infra for Podcast Pile.

## Architecture

The manager server maintains a queue of episodes to process. Worker servers request jobs from the manager, which assigns them tasks. Workers then process their assigned jobs and return results (transcription, diarization, Internet Archive URL, etc.) to the manager, which stores them in the database.

Jobs automatically expire after 2 hours if not completed. Failed jobs are reassigned, but if the URL is unavailable (404, etc.), the job is marked as failed and not reassigned.

Workers download episodes from their remote URLs, perform diarization using Nvidia NeMo, transcribe audio using Parakeet, and send the results back to the manager.

In the background, the manager submits URLs to the Internet Archive's Save Page Now API for archival. The manager only serves URLs from the Internet Archive rather than direct episode URLs (to avoid timestamp issues with dynamic ads).

## Installation

```
pip install -e .
```

## Running

Manager:

```
ppcli manager -p 8000
```

Worker (must have a GPU with 24GB+ of VRAM):

```
ppcli worker -m http://<manager-host>:8000
```