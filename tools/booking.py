"""
Booking Lookup Tool Module.

This module provides the `BookingLookupTool`, an essential component for the AI agent to access
and query the internal synthetic knowledge base of Nawaloka Hospitals' patient appointments. 

By exposing this functionality as a tool, agents can dynamically look up patient bookings 
based on a name, phone number, or email. This file plays a critical role in demonstrating 
the 'Tool Use' and 'ReAct' design patterns, allowing the LLM to cross-reference user queries 
with the hospital's internal JSON database before formulating a response.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def _resolve_bookings_path() -> Path:
    """
    Dynamically resolves the file path to the internal 'bookings.json' database.

    Since the agent can be executed from various directories (e.g., from the root or inside the `rag/` folder),
    this helper function checks multiple common relative paths to ensure the JSON database is always found.

    Returns:
        Path: The resolved absolute or relative path to the bookings.json file.
    """
    cwd = Path.cwd()

    # First, check if the data directory is directly inside the current working directory
    if (cwd / "data" / "bookings.json").exists():
        return cwd / "data" / "bookings.json"

    parent = cwd.parent

    # Second, check if the script is being run from a sub-folder (like 'tools/' or 'rag/'), 
    # making the data directory exist one level up in the parent directory
    if (parent / "data" / "bookings.json").exists():
        return parent / "data" / "bookings.json"

    # Fallback to assuming it's in the current directory's data folder, 
    # which will allow downstream errors to explicitly state it's missing from the expected location
    return cwd / "data" / "bookings.json"


class BookingLookupTool:
    """
    A tool class designed to query the internal hospital bookings JSON database.

    This class handles the lazy-loading of the JSON file, sanitizing search queries, 
    matching patient records based on partial or full matches, and formatting the results 
    into a structured string that an LLM can easily understand.
    """
    
    def __init__(self, bookings_path: Optional[Path] = None):
        """
        Initializes the BookingLookupTool.

        Args:
            bookings_path (Optional[Path]): An explicit path to the bookings.json file. 
                                            If None, the path is resolved automatically.
        """
        self.bookings_path = bookings_path or _resolve_bookings_path()
        self._data: Optional[Dict[str, Any]] = None

    def _load(self) -> Dict[str, Any]:
        """
        Lazily loads the bookings data from the JSON file into memory.

        This method implements lazy loading to ensure the file IO operation only occurs 
        the first time a query is executed, optimizing memory usage and startup time.

        Returns:
            Dict[str, Any]: The loaded JSON data containing patient bookings.
        """
        if self._data is None:
            # If the database doesn't exist, we return an empty skeleton rather than crashing,
            # allowing the AI to gracefully inform the user that no database is available.
            if not self.bookings_path.exists():
                self._data = {"patient_bookings": []}
                return self._data
                
            with open(self.bookings_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
                
        return self._data

    def lookup(
        self,
        patient_name: Optional[str] = None,
        patient_phone: Optional[str] = None,
        patient_email: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches the bookings database for records matching the provided patient details.

        This method performs a flexible "OR" search. If any of the provided search criteria 
        (name, phone, or email) yield a partial or exact match, the booking is retrieved.

        Args:
            patient_name (Optional[str]): The name of the patient to search for.
            patient_phone (Optional[str]): The phone number of the patient.
            patient_email (Optional[str]): The email address of the patient.

        Returns:
            List[Dict[str, Any]]: A list of unique booking records that matched the search criteria.
        """
        data = self._load()
        bookings = data.get("patient_bookings", [])

        if not bookings:
            return []

        # Complex Logic Explanation:
        # String matching can be highly volatile due to case sensitivity and accidental whitespaces.
        # This helper function strictly normalizes all inputs to lowercase and strips whitespace 
        # to ensure that queries like " John Doe " match records like "john doe".
        def norm(s: str) -> str:
            return (s or "").strip().lower()

        name = norm(patient_name)
        phone = norm(patient_phone)
        email = norm(patient_email)

        # Early exit if the LLM called the tool but didn't provide any search parameters
        if not name and not phone and not email:
            return []

        matches = []

        # Iterate through all internal bookings to find partial matches
        for b in bookings:
            b_name = norm(b.get("patient_name", ""))
            b_phone = norm(b.get("patient_phone", ""))
            b_email = norm(b.get("patient_email", ""))

            # We use a bidirectional "in" check. This ensures that if the user searches "John" 
            # it matches "John Doe" (name in b_name), and if they search "John Doe Smith", 
            # it matches "John Doe" (b_name in name).
            if name and (name in b_name or b_name in name):
                matches.append(b)
                continue
            if phone and (phone in b_phone or b_phone in phone):
                matches.append(b)
                continue
            if email and (email in b_email or b_email in email):
                matches.append(b)
                continue

        # Complex Logic Explanation:
        # Because we check multiple criteria independently, a single booking might be appended 
        # to the 'matches' list multiple times (e.g., if both the name AND email match).
        # We use a set ('seen') to track the unique 'booking_id's and filter out any duplicates 
        # before returning the final results to the LLM.
        seen = set()
        unique = []

        for m in matches:
            bid = m.get("booking_id")
            if bid and bid not in seen:
                seen.add(bid)
                unique.append(m)
            elif not bid:
                # Fallback: If a record lacks an ID for some reason, we append it anyway 
                # to prevent data loss, even though it bypasses the deduplication logic.
                unique.append(m)
                
        return unique

    def format_bookings_for_agent(self, bookings: List[Dict[str, Any]]) -> str:
        """
        Formats a list of raw booking dictionaries into a clean, human-readable string.

        LLMs perform much better when structured data (like JSON) is flattened into clear, 
        concise text formats rather than raw brackets and quotes. This method standardizes 
        the output structure.

        Args:
            bookings (List[Dict[str, Any]]): A list of dictionary records returned by the `lookup` method.

        Returns:
            str: A formatted, multi-line string presenting the booking details clearly.
        """
        if not bookings:
            return "No matching bookings found."

        lines = []

        for i, b in enumerate(bookings, 1):
            # Construct a dense but highly readable pipe-delimited summary for each booking record
            lines.append(
                f"Booking {i}: {b.get('booking_id', 'N/A')} | "
                f"Patient: {b.get('patient_name')} | "
                f"Doctor: {b.get('doctor')} | "
                f"Specialty: {b.get('specialty')} | "
                f"Reason: {b.get('reason')} | "
                f"Date: {b.get('date')} at {b.get('time')} | "
                f"Hospital: {b.get('hospital')} | "
                f"Location: {b.get('location')} | "
                f"Status: {b.get('status')} | "
                f"Notes: {b.get('notes', 'None')}"
            )

        return "\n".join(lines)


def get_booking_tool(bookings_path: Optional[Path] = None) -> BookingLookupTool:
    """
    Factory function to retrieve an instance of the BookingLookupTool.

    Args:
        bookings_path (Optional[Path]): An explicit path to the bookings.json file. Defaults to None.

    Returns:
        BookingLookupTool: A ready-to-use instance of the booking lookup tool.
    """
    return BookingLookupTool(bookings_path=bookings_path)