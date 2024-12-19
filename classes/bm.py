import time

class BoyerMoore:
    def _build_bad_character_table(self, pattern):
        """Builds the bad character heuristic table."""
        table = {}
        pattern_length = len(pattern)

        for i in range(pattern_length):
            table[pattern[i]] = i

        return table

    def _build_good_suffix_table(self, pattern):
        """Builds the good suffix heuristic table."""
        pattern_length = len(pattern)
        good_suffix = [-1] * pattern_length
        suffix_length = [0] * pattern_length
        
        for i in range(pattern_length - 1):
            j = i
            k = 0
            while j >= 0 and pattern[j] == pattern[pattern_length - 1 - k]:
                j -= 1
                k += 1
                suffix_length[k] = j + 1

        for i in range(pattern_length - 1):
            j = suffix_length[pattern_length - 1 - i]
            if j != -1:
                good_suffix[j] = pattern_length - 1 - i

        return good_suffix

    def search_pattern(self, text, pattern):
        """Implements the Boyer-Moore string search algorithm."""
        text_length = len(text)
        pattern_length = len(pattern)

        if pattern_length == 0:
            return 0

        bad_char_table = self._build_bad_character_table(pattern)
        good_suffix_table = self._build_good_suffix_table(pattern)

        matches = []
        shift = 0

        while shift <= text_length - pattern_length:
            j = pattern_length - 1

            while j >= 0 and pattern[j] == text[shift + j]:
                j -= 1

            if j < 0:
                matches.append(shift)
                shift += pattern_length - good_suffix_table[0] if pattern_length > 1 else 1
            else:
                bad_char_shift = j - bad_char_table.get(text[shift + j], -1)
                good_suffix_shift = good_suffix_table[j] if j < pattern_length - 1 else 1
                shift += max(bad_char_shift, good_suffix_shift)

        return matches
    
# Example usage
# if __name__ == "__main__":
#     text = "ABAAABCD" * 10000000
#     pattern = "ABC"
#     bm = BoyerMoore()
#     start_time = time.time()
#     result = bm.search_pattern(text, pattern)
#     duration = time.time() - start_time
#     print(f"Time elapse: {duration * 1000} ms")
#     print(f"Total pattern found: {len(result)}")
    