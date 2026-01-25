import re
import numpy as np
import struct

def parse_phoenix_header(file_path):
    """
    Parses the ASCII header of an MSTAR/Phoenix file.
    Extracts: Height, Width, Header Length, and Target Serial Number.
    """
    meta = {}
    try:
        with open(file_path, 'rb') as f:
            # Read enough chunks to cover the header (usually < 2KB)
            header_text = b""
            while True:
                chunk = f.read(1024)
                if not chunk: break
                header_text += chunk
                # Header ends with "EndofPhoenixHeader"
                if b"EndofPhoenixHeader" in header_text:
                    break
            
            # Decode to string (ignoring errors for safety)
            header_str = header_text.decode('utf-8', errors='ignore')
            
            # 1. Extract Dimensions
            rows_match = re.search(r'NumberOfRows\s*=\s*(\d+)', header_str)
            cols_match = re.search(r'NumberOfColumns\s*=\s*(\d+)', header_str)
            
            if rows_match and cols_match:
                meta['h'] = int(rows_match.group(1))
                meta['w'] = int(cols_match.group(1))
            else:
                return None # corrupt header
            
            # 2. Extract Header Length
            # We look for the exact byte length if specified
            len_match = re.search(r'HeaderLength\s*=\s*(\d+)', header_str)
            if len_match:
                meta['header_len'] = int(len_match.group(1))
            else:
                # Fallback: Find the "EndofPhoenixHeader" tag
                end_tag = b"EndofPhoenixHeader"
                idx = header_text.find(end_tag)
                if idx != -1:
                    # Usually followed by a newline (0x0A) or two
                    # We seek to this point + len(tag) + 1
                    meta['header_len'] = idx + len(end_tag) + 1
                else:
                    return None
            
            # 3. Extract Serial Number (Target Type)
            # Useful for debugging class mismatches
            ser_match = re.search(r'TargetSerNum\s*=\s*(\S+)', header_str)
            if ser_match:
                meta['serial_num'] = ser_match.group(1)
            else:
                meta['serial_num'] = "Unknown"
                
    except Exception as e:
        # print(f"Header Parse Error {file_path}: {e}")
        return None
        
    return meta

def read_phoenix_data(file_path, meta):
    """
    Reads the binary payload of an MSTAR file.
    Format: Contiguous 32-bit big-endian floats.
    Standard MSTAR chips are usually (Magnitude, Phase) pairs.
    """
    h, w = meta['h'], meta['w']
    num_pixels = h * w
    
    with open(file_path, 'rb') as f:
        # Skip the ASCII header
        f.seek(meta['header_len'])
        
        # Calculate expected bytes: 2 floats per pixel * 4 bytes
        expected_bytes = num_pixels * 2 * 4
        
        # Read buffer
        buffer = f.read(expected_bytes)
        
        if len(buffer) < expected_bytes:
            # Incomplete file
            return np.zeros((h, w), dtype=np.complex64)

        # Convert to numpy array
        # MSTAR data is Big-Endian (>), Float32 (f4)
        raw_data = np.frombuffer(buffer, dtype='>f4')
        
        # Check integrity
        if raw_data.size != num_pixels * 2:
            return np.zeros((h, w), dtype=np.complex64)
            
        # The data is interleaved: Mag, Phase, Mag, Phase...
        magnitude = raw_data[0::2].reshape(h, w)
        phase = raw_data[1::2].reshape(h, w)
        
        # Convert to Rectangular Complex Form
        # z = r * (cos + j*sin)
        real = magnitude * np.cos(phase)
        imag = magnitude * np.sin(phase)
        
        return real + 1j * imag