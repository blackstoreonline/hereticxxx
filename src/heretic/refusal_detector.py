# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Efficient refusal detection using optimized string matching."""


class RefusalDetector:
    """Optimized refusal detection with optional Aho-Corasick automaton support.
    
    This class provides efficient string matching for refusal markers in model responses.
    It attempts to use the pyahocorasick library for O(n) matching, falling back to
    an optimized linear search if the library is not available.
    
    Performance comparison for 200 responses × 34 markers:
    - Naive approach: O(n×m) = ~6,800 substring checks
    - Aho-Corasick: O(n) = ~200 character scans
    - Optimized fallback: O(n×m) but with early exits and cache optimization
    """

    def __init__(self, refusal_markers: list[str]):
        """Initialize detector with refusal markers.
        
        Args:
            refusal_markers: List of strings that indicate a refusal
        """
        self.refusal_markers = refusal_markers
        self.markers_lower = [marker.lower() for marker in refusal_markers]
        
        # Try to import and use Aho-Corasick for optimal performance
        try:
            import ahocorasick
            self.automaton = ahocorasick.Automaton()
            for marker in self.markers_lower:
                self.automaton.add_word(marker, marker)
            self.automaton.make_automaton()
            self.use_automaton = True
        except ImportError:
            # Fall back to optimized linear search
            self.automaton = None
            self.use_automaton = False
            
            # Sort markers by frequency (most common first for early exit)
            # Common refusal patterns typically start with these
            priority_prefixes = ["i can't", "i cannot", "i'm sorry", "sorry,"]
            self.markers_lower.sort(
                key=lambda m: (
                    not any(m.startswith(p) for p in priority_prefixes),
                    len(m)
                )
            )
    
    def is_refusal(self, response: str) -> bool:
        """Check if a response contains refusal markers.
        
        Args:
            response: The model response to check (should already be normalized)
            
        Returns:
            True if response contains any refusal marker, False otherwise
        """
        if self.use_automaton:
            # O(n) search using Aho-Corasick automaton
            for _ in self.automaton.iter(response):
                return True
            return False
        else:
            # Optimized O(n×m) search with early exit
            for marker in self.markers_lower:
                if marker in response:
                    return True
            return False
