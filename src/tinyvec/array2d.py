import struct
import ctypes
from typing import Tuple, Optional
        
import numpy as np

HEADER_SZ = 12  # 3 integers (4 bytes each) for rows, cols, and precision flag

class Array2DFileHandler:
    filepath: str
    """Path to the file"""
    rows: int
    """Number of rows in the array"""
    cols: int
    """Number of columns in the array"""
    precision: str
    """Precision of the floating point values: 'single' or 'double'"""
    struct_fmt: str
    """Struct format string for packing/unpacking the floating point values"""
    float_size: int
    """Size of a single floating point value in bytes"""
    header_fmt: str
    """Struct format string for packing/unpacking the header values"""


    def __init__(self, filepath: str, rows: int = 0, cols: int = 0, precision: str = 'single'):
        self.filepath = filepath
        self.rows = rows
        self.cols = cols
        self.precision = precision
        self.struct_fmt = 'f' if precision == 'single' else 'd'
        self.float_size = ctypes.sizeof(ctypes.c_float) if precision == 'single' else ctypes.sizeof(ctypes.c_double)
        self.header_fmt = '3i'  # rows, cols, flags
        # Initialize or read the file header
        self._init_or_read_header()

    def _init_or_read_header(self):
        try:
            with open(self.filepath, 'r+b') as file:
                self._parse_header(file)
        except FileNotFoundError:
            self._create_file()

    def _create_file(self):
        flags = 0b01 if self.precision == 'double' else 0b00  # Set precision flag, leaving lock bit as 0
        with open(self.filepath, 'wb') as file:
            file.write(struct.pack(self.header_fmt, self.rows, self.cols, flags))

    def _parse_header(self, file):
        file.seek(0)
        self.rows, self.cols, flags = struct.unpack(self.header_fmt, file.read(HEADER_SZ))
        self.precision = 'double' if flags & 0b01 else 'single'
        self.struct_fmt = 'd' if self.precision == 'double' else 'f'
        # Check if the file is locked
        if flags & 0b10:
            raise Exception("File is currently locked for writing.")

    def _set_lock(self, file, lock: bool):
        # Assume file is an open file object with 'r+b' mode
        file.seek(8)  # Seek directly to the flags part of the header
        flags, = struct.unpack('i', file.read(4))
        if lock:
            flags |= 0b10  # Set lock flag
        else:
            flags &= ~0b10  # Clear lock flag
        file.seek(8)  # Seek back to the flags part of the header to update it
        file.write(struct.pack('i', flags))
        file.flush()


    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, 'rb') as file:
            rows, cols, precision_flag = struct.unpack('3i', file.read(HEADER_SZ))
            precision = 'single' if precision_flag == 0 else 'double'
            return cls(filepath, rows, cols, precision)

    def read(self) -> np.ndarray:
        """Read the entire dataset into a NumPy array.

        Returns:
            np.ndarray: The dataset as a NumPy array with shape (rows, cols).
        """
        with open(self.filepath, 'rb') as file:
            self._parse_header(file)  # Ensure no write lock is active and update attributes
            # Calculate the total size of the dataset to read (excluding the header)
            data_size = self.float_size * self.cols * self.rows
            # Seek to the start of the data block, right after the header
            file.seek(HEADER_SZ)
            # Read the entire data block
            data_block = file.read(data_size)
            
        # Determine the appropriate NumPy dtype based on the struct format
        np_dtype = np.float32 if self.struct_fmt == 'f' else np.float64
        
        # Use numpy.frombuffer to directly create a NumPy array from the binary data
        data_array = np.frombuffer(data_block, dtype=np_dtype)
        
        # Reshape the array to have the correct dimensions (self.rows x self.cols)
        data_array = data_array.reshape((self.rows, self.cols))
        
        return data_array


    def seek_row(self, row_number: int) -> list[float]:
        with open(self.filepath, 'rb') as file:
            self._parse_header(file)  # Ensure no write lock is active and update attributes
            offset = HEADER_SZ + self.float_size * self.cols * row_number
            file.seek(offset)
            row_data = [struct.unpack(self.struct_fmt, file.read(self.float_size))[0] for _ in range(self.cols)]
            return row_data


    def load_batch(self, start_row: int, end_row: int) -> np.ndarray:
        """Load a batch of rows from the file directly into a NumPy array.

        Args:
            start_row (int): The starting row index of the batch.
            end_row (int): The ending row index of the batch (exclusive).

        Returns:
            np.ndarray: The batch of rows as a NumPy array.
        """
        with open(self.filepath, 'rb') as file:
            self._parse_header(file)  # Ensure no write lock is active and update attributes
            # Calculate the offset for the start of the batch and seek to it
            start_offset = 12 + self.float_size * self.cols * start_row
            file.seek(start_offset)
            # Calculate the number of rows and the total size of the batch to read
            num_rows = min(end_row, self.rows) - start_row
            batch_size = self.float_size * self.cols * num_rows
            # Read the entire batch data
            batch_data = file.read(batch_size)
            
        # Determine the appropriate NumPy dtype based on the struct format
        np_dtype = np.float32 if self.struct_fmt == 'f' else np.float64
        
        # Use numpy.frombuffer to directly create a NumPy array from the binary data
        batch_array = np.frombuffer(batch_data, dtype=np_dtype)
        
        # Reshape the array to have the correct dimensions (num_rows x self.cols)
        batch_array = batch_array.reshape((num_rows, self.cols))
        
        return batch_array

        
    def write(self, data: list[list[float]]):
        with open(self.filepath, 'r+b') as file:
            self._parse_header(file)  # Ensures we have the latest file info and no write is currently happening
            # Set the write lock with the current file object
            self._set_lock(file, True)
            try:
                # Move to the end of the file for appending data
                file.seek(0, 2)
                for row in data:
                    for value in row:
                        file.write(struct.pack(self.struct_fmt, value))
                
                # Update rows count in the header
                self.rows += len(data)
                # It's important to move back to the start to update the header after appending data
                file.seek(0)
                flags = 0b01 if self.precision == 'double' else 0b00  # Update flags if necessary
                # Now include the flags in the header update to reflect potential changes
                file.write(struct.pack(self.header_fmt, self.rows, self.cols, flags))
            finally:
                # Clear the write lock with the current file object
                self._set_lock(file, False)


    def write_row(self, row_data: list[float], row_number: Optional[int] = None):
        with open(self.filepath, 'r+b') as file:
            self._parse_header(file)  # Read and update attributes to ensure consistency
            
            if row_number is not None:
                if row_number < 0 or row_number >= self.rows:
                    raise ValueError("row_number out of range")
                offset = HEADER_SZ + self.float_size * self.cols * row_number
            else:
                # Calculate offset for appending; actual row increment and header update happens after lock is set
                offset = HEADER_SZ + self.float_size * self.cols * self.rows
            
            self._set_lock(file, True)  # Set the write lock after successfully parsing header and before modifying the file
            try:
                if row_number is None:
                    # Append mode, now safely increase row count and update header
                    self.rows += 1
                    file.seek(0)
                    flags = 0b01 if self.precision == 'double' else 0b00  # Optionally recalculate flags, if needed
                    file.write(struct.pack(self.header_fmt, self.rows, self.cols, flags))
                
                file.seek(offset)
                for value in row_data:
                    file.write(struct.pack(self.struct_fmt, value))
            finally:
                self._set_lock(file, False)  # Ensure the lock is cleared even if an error occurs

    def delete_row(self, row_number: int) -> Tuple[int, int]:
        """
        Deletes a row from the file and moves the last row to the deleted row's position if necessary.

        Args:
            row_number (int): The index of the row to delete.

        Returns:
            Tuple[int, int]: The index of the moved row and its new location, if any row was moved. Otherwise, returns None.
        """
        if row_number < 0 or row_number >= self.rows:
            raise ValueError("row_number out of range")

        moved_row_index = None
        new_location = None
        with open(self.filepath, 'r+b') as file:
            self._parse_header(file)  # Ensure no write lock is active and update attributes
            self._set_lock(file, True)
            try:
                if row_number != self.rows - 1:  # If not deleting the last row
                    # Move the last row to the position of the deleted row
                    moved_row_index = self.rows - 1
                    new_location = row_number
                    # Read the last row's data
                    offset = HEADER_SZ + self.float_size * self.cols * (self.rows - 1)
                    file.seek(offset)
                    row_data = [struct.unpack(self.struct_fmt, file.read(self.float_size))[0] for _ in range(self.cols)]
                    # Write the last row's data to the deleted row's position
                    offset = HEADER_SZ + self.float_size * self.cols * new_location
                    file.seek(offset)
                    for value in row_data:
                        file.write(struct.pack(self.struct_fmt, value))

                # Truncate the file to remove the last row
                new_size = HEADER_SZ + self.float_size * self.cols * (self.rows - 1)
                file.truncate(new_size)
                self.rows -= 1  # Update rows count

                # Update the header with the new rows count
                file.seek(0)
                flags = 0b01 if self.precision == 'double' else 0b00
                file.write(struct.pack(self.header_fmt, self.rows, self.cols, flags))
            finally:
                self._set_lock(file, False)

        return moved_row_index, new_location
