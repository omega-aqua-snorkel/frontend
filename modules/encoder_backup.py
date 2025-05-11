import logging
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple

import ffmpeg
from langcodes import Language
from pydantic import BaseModel, computed_field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
WHITELIST_LANGUAGES = ["eng", "hin", "ben"]
BITRATE_SETTINGS = {
    1080: {"bitrate": "5000k", "maxrate": "5350k", "bufsize": "7500k"},
    720: {"bitrate": "2800k", "maxrate": "2996k", "bufsize": "4200k"},
    480: {"bitrate": "1400k", "maxrate": "1498k", "bufsize": "2100k"},
    360: {"bitrate": "800k", "maxrate": "856k", "bufsize": "1200k"},
    240: {"bitrate": "400k", "maxrate": "428k", "bufsize": "600k"},
}


class StreamType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"


class Stream(BaseModel):
    index: int
    type: StreamType
    codec_name: str
    language: str = "und"
    source_file: Optional[Path] = None
    file: Optional[Path] = None
    raw: Dict[str, Any]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    @computed_field
    @property
    def name(self) -> str:
        return (
            f"stream-{self.index}-{self.type.value}-{self.codec_name}-{self.language}"
        )

    @computed_field
    @property
    def display_language(self) -> str:
        return Language.get(self.language).display_name()

    @computed_field
    @property
    def stream_ext(self) -> str:
        if self.type == StreamType.SUBTITLE:
            return ".vtt" if self.codec_name in ["subrip", "ass"] else ".srt"
        return ".mp4"

    @computed_field
    @property
    def is_subtitle(self) -> bool:
        return self.type == StreamType.SUBTITLE

    @computed_field
    @property
    def is_video(self) -> bool:
        return self.type == StreamType.VIDEO

    @computed_field
    @property
    def is_audio(self) -> bool:
        return self.type == StreamType.AUDIO

    @computed_field
    @property
    def stream_dir_name(self) -> str:
        return f"{self.type.value}_{self.index}_{self.language}"

    def extract(
        self,
        output_file: Path,
        source_file: Optional[Path] = None,
        encoding_params: Optional[dict] = None,
    ) -> Path:
        if not self.source_file and not source_file:
            raise Exception(
                "Source file not provided, and stream does not have a source file"
            )

        output_file = Path(output_file).absolute()
        stream_input = ffmpeg.input(str(source_file or self.source_file))

        params = {
            "map": f"0:{self.index}",
        }

        if self.type != StreamType.SUBTITLE or self.codec_name not in ["subrip", "ass"]:
            params["c"] = "copy"

        if encoding_params:
            params.update(encoding_params)

        # Copy stream without re-encoding
        logger.debug(f"Extracting {self.name} to {output_file} with params: {params}")
        stream_output = ffmpeg.output(stream_input, str(output_file), **params)

        ffmpeg.run(stream_output, quiet=True, overwrite_output=True)
        self.file = output_file
        return self.file

    def change_resolution(
        self,
        resolution: Tuple[int, int],
        output_file: Path,
        encoding_params: Optional[dict] = None,
    ):
        if self.type != StreamType.VIDEO:
            raise Exception("Stream is not a video")

        stream_input = ffmpeg.input(str(self.file))

        params = {
            "vf": f"scale={resolution[0]}:{resolution[1]}",
            "c": "libx264",
            "preset": "ultrafast",
            "crf": 23,
            "sws_flags": "fast_bilinear",
        }

        if encoding_params:
            params.update(encoding_params)

        stream_output = ffmpeg.output(stream_input, str(output_file), **params)

        logger.debug(f"Transcoding {self.name} to {output_file} with params: {params}")
        ffmpeg.run(stream_output, quiet=True, overwrite_output=True)
        self.file = output_file
        return self.file

    def create_hls_playlist(
        self,
        output_dir: Path,
        segment_duration: int = 10,
        resolution: Optional[int] = None,
        bitrate_info: Optional[Dict[str, str]] = None,
        copy_codec: bool = True,
    ) -> Optional[Path]:
        """
        Create HLS playlist for this stream

        Args:
            output_dir: Directory to store HLS output
            segment_duration: Duration of each segment in seconds
            resolution: Target resolution height (only for video)
            bitrate_info: Bitrate settings for video encoding
            copy_codec: Whether to copy codec without transcoding

        Returns:
            Path to the playlist file or None if failed
        """
        try:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            if self.is_subtitle and self.file and self.file.suffix == ".vtt":
                return self._create_subtitle_hls(output_dir, segment_duration)
            elif self.is_audio:
                return self._create_audio_hls(output_dir, segment_duration)
            elif self.is_video:
                return self._create_video_hls(
                    output_dir, segment_duration, resolution, bitrate_info, copy_codec
                )

            return None
        except Exception as e:
            logger.warning(f"Failed to create HLS playlist for {self.name}: {e}")
            return None

    def _create_subtitle_hls(
        self, output_dir: Path, segment_duration: int
    ) -> Optional[Path]:
        """Create HLS playlist for subtitle stream"""
        try:
            if not self.file or not self.file.exists():
                logger.warning(f"Subtitle file not found: {self.file}")
                return None

            dest_file = output_dir / "subtitles.vtt"

            # Read the original VTT file
            with open(self.file, "r") as src:
                content = src.read()

            # Add X-TIMESTAMP-MAP after WEBVTT line if not already present
            if "X-TIMESTAMP-MAP" not in content:
                content = content.replace(
                    "WEBVTT", "WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:900000,LOCAL:00:00:00.000"
                )

            # Write the modified content to the destination file
            with open(dest_file, "w") as dst:
                dst.write(content)

            # Get VTT file size for byte range calculations
            file_size = os.path.getsize(dest_file)

            # Create a playlist using byte ranges
            playlist_file = output_dir / "main.m3u8"

            duration = float(
                self.raw.get("duration", segment_duration * 10)
            )  # Default duration if not available

            with open(playlist_file, "w") as f:
                f.write("#EXTM3U\n")
                f.write("#EXT-X-VERSION:6\n")
                f.write(f"#EXT-X-TARGETDURATION:{int(duration)}\n")
                f.write("#EXT-X-PLAYLIST-TYPE:VOD\n")

                f.write(f"#EXTINF:{duration},\n")
                f.write(f"#EXT-X-BYTERANGE:{file_size}@0\n")
                f.write(f"subtitles.vtt\n")
                f.write("#EXT-X-ENDLIST\n")

            logger.info(
                f"Created subtitle playlist with byte ranges for stream {self.index} ({self.language})"
            )
            return playlist_file

        except Exception as e:
            logger.warning(f"Failed to create subtitle HLS playlist: {e}")
            return None

    def _create_audio_hls(
        self, output_dir: Path, segment_duration: int
    ) -> Optional[Path]:
        """Create HLS playlist for audio stream"""
        try:
            if not self.file or not self.file.exists():
                logger.warning(f"Audio file not found: {self.file}")
                return None

            cmd = [
                "ffmpeg",
                "-i",
                str(self.file),
                "-c:a",
                "copy",  # Copy audio codec without re-encoding
                "-f",
                "hls",
                "-hls_time",
                str(segment_duration),
                "-hls_playlist_type",
                "vod",
                "-hls_flags",
                "independent_segments",
                "-hls_segment_type",
                "mpegts",
                "-hls_segment_filename",
                f"{output_dir}/%03d.ts",
                f"{output_dir}/main.m3u8",
            ]

            logger.info(
                f"Creating HLS segments for audio stream {self.index} ({self.language}) with direct copying"
            )
            subprocess.run(cmd, check=True)

            return output_dir / "main.m3u8"
        except Exception as e:
            logger.warning(f"Failed to create audio HLS playlist: {e}")
            return None

    def _create_video_hls(
        self,
        output_dir: Path,
        segment_duration: int,
        resolution: Optional[int] = None,
        bitrate_info: Optional[Dict[str, str]] = None,
        copy_codec: bool = True,
    ) -> Optional[Path]:
        """Create HLS playlist for video stream"""
        try:
            if not self.file or not self.file.exists():
                logger.warning(f"Video file not found: {self.file}")
                return None

            if copy_codec:
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(self.file),
                    "-c:v",
                    "copy",
                    "-f",
                    "hls",
                    "-hls_time",
                    str(segment_duration),
                    "-hls_playlist_type",
                    "vod",
                    "-hls_flags",
                    "independent_segments",
                    "-hls_segment_type",
                    "mpegts",
                    "-hls_segment_filename",
                    f"{output_dir}/%03d.ts",
                    f"{output_dir}/main.m3u8",
                ]
                logger.info(f"Creating HLS segments for video without re-encoding")
            else:
                if not bitrate_info or not resolution:
                    raise ValueError(
                        "Bitrate info and resolution required for transcoding"
                    )

                cmd = [
                    "ffmpeg",
                    "-i",
                    str(self.file),
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "baseline",
                    "-level",
                    "3.0",
                    "-b:v",
                    bitrate_info["bitrate"],
                    "-maxrate",
                    bitrate_info["maxrate"],
                    "-bufsize",
                    bitrate_info["bufsize"],
                    "-vf",
                    f"scale=-2:{resolution}",
                    "-f",
                    "hls",
                    "-hls_time",
                    str(segment_duration),
                    "-hls_playlist_type",
                    "vod",
                    "-hls_flags",
                    "independent_segments",
                    "-hls_segment_type",
                    "mpegts",
                    "-hls_segment_filename",
                    f"{output_dir}/%03d.ts",
                    f"{output_dir}/main.m3u8",
                ]
                logger.info(
                    f"Creating HLS segments for video with transcoding to {resolution}p"
                )

            subprocess.run(cmd, check=True)
            return output_dir / "main.m3u8"
        except Exception as e:
            logger.warning(f"Failed to create video HLS playlist: {e}")
            return None

    @classmethod
    def from_raw(cls, raw: Dict[str, Any], source_file: Optional[Path] = None) -> Self:
        return cls(
            index=raw["index"],
            type=StreamType(raw["codec_type"]),
            codec_name=raw["codec_name"],
            language=raw.get("tags", {}).get("language", "und"),
            source_file=source_file,
            raw=raw,
        )


class StreamInfo(BaseModel):
    streams: List[Stream]
    format_info: dict
    file: Path

    @computed_field
    @property
    def has_video_stream(self) -> bool:
        return any(stream.type == StreamType.VIDEO for stream in self.streams)

    @computed_field
    @property
    def has_audio_stream(self) -> bool:
        return any(stream.type == StreamType.AUDIO for stream in self.streams)

    @computed_field
    @property
    def has_subtitle_stream(self) -> bool:
        return any(stream.type == StreamType.SUBTITLE for stream in self.streams)

    @computed_field
    @property
    def is_valid(self) -> bool:
        return self.has_video_stream and self.has_audio_stream

    @computed_field
    @property
    def duration(self) -> float:
        """Get the duration of the media file"""
        return float(self.format_info.get("duration", 0))

    @property
    def video_streams(self) -> List[Stream]:
        """Get all video streams"""
        return [s for s in self.streams if s.type == StreamType.VIDEO]

    @property
    def audio_streams(self) -> List[Stream]:
        """Get all audio streams"""
        return [s for s in self.streams if s.type == StreamType.AUDIO]

    @property
    def subtitle_streams(self) -> List[Stream]:
        """Get all subtitle streams"""
        return [s for s in self.streams if s.type == StreamType.SUBTITLE]

    def filter_streams_by_language(self, languages: List[str]) -> "StreamInfo":
        """Filter streams by language whitelist"""
        filtered_streams = [
            s
            for s in self.streams
            if s.type == StreamType.VIDEO or s.language in languages
        ]
        return StreamInfo(
            streams=filtered_streams, format_info=self.format_info, file=self.file
        )

    def extract_all_streams(self, output_dir: Path) -> "StreamInfo":
        """Extract all streams to separate files"""
        extracted_streams = []

        for stream in self.streams:
            try:
                ext = stream.stream_ext
                output_file = output_dir / f"{stream.name}{ext}"
                stream.extract(output_file)
                extracted_streams.append(stream)
            except Exception as e:
                logger.warning(f"Failed to extract stream {stream.name}: {e}")

        return StreamInfo(
            streams=extracted_streams, format_info=self.format_info, file=self.file
        )

    @classmethod
    def from_file(cls, file: Path) -> Self:
        file = Path(file).absolute()

        try:
            probe = ffmpeg.probe(str(file))
        except ffmpeg.Error as e:
            raise Exception(
                f"Error probing input file: {e.stderr.decode() if e.stderr else str(e)}"
            )

        streams = probe["streams"]
        formatted_streams = []
        for stream in streams:
            try:
                formatted_streams.append(Stream.from_raw(stream, file))
            except Exception as e:
                logger.warning(f"Skipping stream {stream['index']}: {e}")
        return cls(streams=formatted_streams, format_info=probe["format"], file=file)


class HLSStreamDescriptor(BaseModel):
    """Represents a stream in the HLS playlist"""

    type: StreamType
    index: int
    language: str
    language_name: str
    playlist: Path
    bandwidth: int
    resolution: Optional[int] = None

    @computed_field
    @property
    def relative_playlist(self) -> str:
        """Get the relative path to the master playlist"""
        return str(self.playlist)

    @computed_field
    @property
    def width(self) -> Optional[int]:
        """Calculate width based on 16:9 aspect ratio if resolution is provided"""
        if self.resolution:
            return int(self.resolution * 16 / 9)
        return None


class HLSEncoder(BaseModel):
    """HLS encoder configuration and methods"""

    stream_info: StreamInfo
    output_dir: Path
    segment_duration: int = 10
    resolutions: Optional[List[int]] = None
    language_whitelist: List[str] = WHITELIST_LANGUAGES

    def encode(self) -> Path:
        """
        Encode the input file to HLS format

        Returns:
            Path to the master playlist file
        """
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        streams_dir = self.output_dir / "streams"
        os.makedirs(streams_dir, exist_ok=True)
        logger.info(f"Created temporary directory for extracted streams: {streams_dir}")

        # Filter streams by language whitelist
        filtered_info = self.stream_info.filter_streams_by_language(
            self.language_whitelist
        )

        # Extract all streams
        extracted_info = filtered_info.extract_all_streams(streams_dir)

        # Process streams into HLS
        video_descriptors = self._process_video_streams(extracted_info.video_streams)
        audio_descriptors = self._process_audio_streams(extracted_info.audio_streams)
        subtitle_descriptors = self._process_subtitle_streams(
            extracted_info.subtitle_streams
        )

        # Create master playlist
        return self._create_master_playlist(
            video_descriptors, audio_descriptors, subtitle_descriptors
        )

    def _process_video_streams(
        self, video_streams: List[Stream]
    ) -> List[HLSStreamDescriptor]:
        """Process video streams into HLS"""
        descriptors = []

        for stream in video_streams:
            # Get source video resolution
            try:
                if stream.file:
                    video_probe = ffmpeg.probe(str(stream.file))
                    source_height = int(video_probe["streams"][0]["height"])
                    logger.info(f"Source video height: {source_height}p")
                else:
                    raise ValueError("Video file not available")
            except Exception as e:
                logger.warning(f"Failed to get source resolution: {e}")
                source_height = 1080  # Default to 1080p if we can't determine

            # If resolutions is None/empty, just copy the source video
            if not self.resolutions:
                stream_output_dir = self.output_dir / stream.stream_dir_name

                # Get default bitrate for this resolution
                bitrate_info = BITRATE_SETTINGS.get(
                    source_height,
                    {"bitrate": "2000k", "maxrate": "2140k", "bufsize": "3000k"},
                )

                playlist = stream.create_hls_playlist(
                    stream_output_dir, self.segment_duration, copy_codec=True
                )

                if playlist:
                    descriptors.append(
                        HLSStreamDescriptor(
                            type=stream.type,
                            index=stream.index,
                            language=stream.language,
                            language_name=stream.display_language,
                            resolution=source_height,
                            playlist=playlist,
                            bandwidth=int(bitrate_info["bitrate"].replace("k", "000")),
                        )
                    )

                continue

            # Process each target resolution
            for resolution in self.resolutions:
                # Skip if desired resolution is higher than source
                if resolution > source_height:
                    logger.info(
                        f"Skipping {resolution}p as it's higher than source resolution ({source_height}p)"
                    )
                    continue

                stream_output_dir = (
                    self.output_dir / f"{stream.stream_dir_name}_{resolution}p"
                )

                # Get bitrate settings for this resolution
                bitrate_info = BITRATE_SETTINGS.get(
                    resolution,
                    {"bitrate": "1000k", "maxrate": "1070k", "bufsize": "1500k"},
                )

                # Check if source resolution is within 100p of target resolution
                if abs(source_height - resolution) <= 100:
                    # Use source file without re-encoding
                    playlist = stream.create_hls_playlist(
                        stream_output_dir, self.segment_duration, copy_codec=True
                    )

                    if playlist:
                        descriptors.append(
                            HLSStreamDescriptor(
                                type=stream.type,
                                index=stream.index,
                                language=stream.language,
                                language_name=stream.display_language,
                                resolution=source_height,
                                playlist=playlist,
                                bandwidth=int(
                                    bitrate_info["bitrate"].replace("k", "000")
                                ),
                            )
                        )
                else:
                    # Create HLS segments with transcoding
                    playlist = stream.create_hls_playlist(
                        stream_output_dir,
                        self.segment_duration,
                        resolution=resolution,
                        bitrate_info=bitrate_info,
                        copy_codec=False,
                    )

                    if playlist:
                        descriptors.append(
                            HLSStreamDescriptor(
                                type=stream.type,
                                index=stream.index,
                                language=stream.language,
                                language_name=stream.display_language,
                                resolution=resolution,
                                playlist=playlist,
                                bandwidth=int(
                                    bitrate_info["bitrate"].replace("k", "000")
                                ),
                            )
                        )

        return descriptors

    def _process_audio_streams(
        self, audio_streams: List[Stream]
    ) -> List[HLSStreamDescriptor]:
        """Process audio streams into HLS"""
        descriptors = []

        for stream in audio_streams:
            stream_output_dir = self.output_dir / stream.stream_dir_name

            playlist = stream.create_hls_playlist(
                stream_output_dir, self.segment_duration
            )

            if playlist:
                descriptors.append(
                    HLSStreamDescriptor(
                        type=stream.type,
                        index=stream.index,
                        language=stream.language,
                        language_name=stream.display_language,
                        playlist=playlist,
                        bandwidth=128000,  # Use a default bandwidth value
                    )
                )

        return descriptors

    def _process_subtitle_streams(
        self, subtitle_streams: List[Stream]
    ) -> List[HLSStreamDescriptor]:
        """Process subtitle streams into HLS"""
        descriptors = []

        for stream in subtitle_streams:
            if stream.file and stream.file.suffix == ".vtt":
                stream_output_dir = self.output_dir / stream.stream_dir_name

                playlist = stream.create_hls_playlist(
                    stream_output_dir, self.segment_duration
                )

                if playlist:
                    descriptors.append(
                        HLSStreamDescriptor(
                            type=stream.type,
                            index=stream.index,
                            language=stream.language,
                            language_name=stream.display_language,
                            playlist=playlist,
                            bandwidth=1000,  # Minimal bandwidth for subtitles
                        )
                    )

        return descriptors

    def _create_master_playlist(
        self,
        video_descriptors: List[HLSStreamDescriptor],
        audio_descriptors: List[HLSStreamDescriptor],
        subtitle_descriptors: List[HLSStreamDescriptor],
    ) -> Path:
        """Create master HLS playlist"""
        master_playlist_path = self.output_dir / "master.m3u8"
        with open(master_playlist_path, "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:6\n\n")

            # Add audio streams
            for audio in audio_descriptors:
                f.write(
                    f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="{audio.language_name}",LANGUAGE="{audio.language}",URI="{os.path.relpath(audio.playlist, self.output_dir)}",DEFAULT=YES,AUTOSELECT=YES\n'
                )

            f.write("\n")

            # Add subtitle streams
            for subtitle in subtitle_descriptors:
                f.write(
                    f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="text",NAME="{subtitle.language_name}",LANGUAGE="{subtitle.language}",URI="{os.path.relpath(subtitle.playlist, self.output_dir)}",DEFAULT=NO,AUTOSELECT=NO\n'
                )

            f.write("\n")

            # Add video streams with different resolutions
            for video in video_descriptors:
                # Include both audio and subtitle groups
                f.write(
                    f'#EXT-X-STREAM-INF:BANDWIDTH={video.bandwidth},RESOLUTION={video.width}x{video.resolution},AUDIO="audio",SUBTITLES="text"\n'
                )
                f.write(f"{os.path.relpath(video.playlist, self.output_dir)}\n")

                f.write("\n")

                # Add I-frame playlist reference
                f.write(
                    f'#EXT-X-I-FRAME-STREAM-INF:BANDWIDTH={video.bandwidth},RESOLUTION={video.width}x{video.resolution},URI="{os.path.relpath(video.playlist, self.output_dir)}"\n'
                )

        logger.info(f"Created master playlist at {master_playlist_path}")
        return master_playlist_path


def encode_to_hls(
    input_file: str,
    output_dir: str,
    resolutions: Optional[list] = None,
    segment_duration: int = 10,
) -> str:
    """
    Encode a video file to HLS format with multiple resolutions using ffmpeg.

    Args:
        input_file: Path to the input video file
        output_dir: Directory to store the HLS output
        resolutions: List of resolutions to encode (default: None) or None to keep source resolution
        segment_duration: Duration of each segment in seconds (default: 10)

    Returns:
        Path to the master playlist file
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)

    # Analyze the input file
    stream_info = StreamInfo.from_file(input_path)

    if not stream_info.is_valid:
        raise ValueError("Input file must have at least one video and one audio stream")

    # Create HLS encoder and encode
    encoder = HLSEncoder(
        stream_info=stream_info,
        output_dir=output_path,
        segment_duration=segment_duration,
        resolutions=resolutions,
    )

    return str(encoder.encode())
