import time

class WuManber:
    def __init__(self, patterns):
        self.patterns = patterns
        self.patterns_len = len(patterns)
        self.block_size = 8  # Size of block, can be adjusted
        self.shift_table = self.build_shift_table(patterns)

    def build_shift_table(self, patterns):
        """
        Construct a shift table to optimize the search.
        """
        shift_table = {}
        for pattern in patterns:
            first_char = pattern[0]  # First character of the pattern
            if first_char not in shift_table:
                shift_table[first_char] = []
            shift_table[first_char].append(pattern)
        return shift_table

    def search(self, text):
        """
        Perform the Wu-Manber search for all patterns in the text.
        """
        matches = []
        n = len(text)
        i = 0

        while i <= n - self.block_size:
            block = text[i:i + self.block_size]
            first_char = block[0]  # Take the first character of the block

            if first_char in self.shift_table:
                # Check each pattern in the corresponding group
                for pattern in self.shift_table[first_char]:
                    if block.startswith(pattern):
                        matches.append((i, pattern))

            # Shift the window by 1 for the next comparison
            i += 1

        return matches

class AhoCorasick:
    def __init__(self, patterns):
        self.num_nodes = 1  # Trie starts with the root node
        self.edges = [{}]  # List of dictionaries for each node's outgoing edges
        self.fail = [-1]  # Failure links for each node
        self.output = [{}]  # Output for each node, to store patterns ending at that node
        
        # Insert patterns into the trie
        for pattern in patterns:
            self.add_pattern(pattern)
        
        # Build the failure links using BFS
        self.build_failure_links()

    def add_pattern(self, pattern):
        current_node = 0
        for char in pattern:
            # If the character is not in the current node's edges, create a new node
            if char not in self.edges[current_node]:
                self.edges[current_node][char] = self.num_nodes
                self.edges.append({})
                self.fail.append(-1)
                self.output.append({})
                self.num_nodes += 1
            # Move to the next node
            current_node = self.edges[current_node][char]
        # Add the pattern to the output of the final node
        self.output[current_node][pattern] = True

    def build_failure_links(self):
        from collections import deque
        queue = deque()

        # Initialize the BFS queue with the children of the root node
        for char in range(256):  # ASCII range (assuming all characters are ASCII)
            char = chr(char)
            if char in self.edges[0]:
                node = self.edges[0][char]
                self.fail[node] = 0  # The fail link of the direct children of the root is 0
                queue.append(node)
            else:
                self.edges[0][char] = 0  # Link to the root node if no match

        # Perform BFS to build failure links for all nodes
        while queue:
            current_node = queue.popleft()
            
            for char, next_node in self.edges[current_node].items():
                queue.append(next_node)
                fail_node = self.fail[current_node]
                
                # Traverse failure links to find the longest suffix that matches the current character
                while char not in self.edges[fail_node]:
                    fail_node = self.fail[fail_node]
                self.fail[next_node] = self.edges[fail_node][char]
                
                # Merge output from failure node to current node's output
                self.output[next_node].update(self.output[self.fail[next_node]])

    def search(self, text):
        current_node = 0
        matches = []

        # Iterate through the text
        for i in range(len(text)):
            char = text[i]

            # Follow failure links until a matching character is found
            while char not in self.edges[current_node]:
                current_node = self.fail[current_node]

            # Move to the next node
            current_node = self.edges[current_node][char]

            # If there's any pattern ending at this node, we have found a match
            if self.output[current_node]:
                for pattern in self.output[current_node]:
                    matches.append((pattern, i - len(pattern) + 1, i))  # Pattern, start index, end index
        
        return matches

# Boyer-Moore Algorithm for Pattern Matching
def boyer_moore_search(text, patterns):
    def boyer_moore(lst, pattern):
        n, m = len(lst), len(pattern)
        bad_char = {ch: m - i - 1 for i, ch in enumerate(pattern)}
        i = 0
        matches = []
        
        while i <= n - m:
            j = m - 1
            while j >= 0 and lst[i + j] == pattern[j]:
                j -= 1
            if j < 0:
                matches.append(i)
                i += m  # Skip the matched part (because of the bad_char rule)
            else:
                i += bad_char.get(lst[i + j], m)  # Skip based on the bad character
        return matches

    all_matches = []
    for pattern in patterns:
        matches = boyer_moore(text, pattern)
        all_matches.append((pattern, matches))
    return all_matches

class TrieNode:
    def __init__(self):
        self.children = {}
        self.fail = None  # Failure link for Aho-Corasick
        self.output = 0   # Bitmask for patterns that end at this node

class TrieBitwiseIndexing:
    def __init__(self, patterns):
        self.root = TrieNode()
        self.patterns = patterns
        self.num_patterns = len(patterns)
        self.build_trie(patterns)
        self.build_failure_links()

    def build_trie(self, patterns):
        """
        Build the Trie structure and associate each pattern with a bitmask.
        """
        for idx, pattern in enumerate(patterns):
            node = self.root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.output |= (1 << idx)  # Set the bit corresponding to this pattern

    def build_failure_links(self):
        """
        Build failure links for the Aho-Corasick-like algorithm using BFS.
        """
        from collections import deque
        queue = deque()

        # Initialize the queue with the children of the root node
        for char, child_node in self.root.children.items():
            child_node.fail = self.root
            queue.append(child_node)

        # BFS to build failure links
        while queue:
            current_node = queue.popleft()
            for char, child_node in current_node.children.items():
                # Traverse the failure links to find the longest suffix
                fail_node = current_node.fail
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail
                if fail_node is None:
                    child_node.fail = self.root
                else:
                    child_node.fail = fail_node.children[char]
                    # Merge the output bitmasks
                    child_node.output |= child_node.fail.output
                queue.append(child_node)

    def search(self, text):
        """
        Perform the search for all patterns in the text using the Trie with Bitwise Indexing.
        """
        node = self.root
        matches = []

        for i in range(len(text)):
            char = text[i]

            # Follow the failure links if the character isn't found
            while node is not None and char not in node.children:
                node = node.fail

            if node is None:
                node = self.root
                continue

            node = node.children[char]

            # Check the output bitmask for matches
            if node.output:
                # Check each bit in the output bitmask to see which patterns matched
                for idx in range(self.num_patterns):
                    if node.output & (1 << idx):  # If the bit is set
                        matches.append((self.patterns[idx], i - len(self.patterns[idx]) + 1, i))

        return matches

# Create a long random text and multiple patterns
text = "123455124" * 10000000  # Long text, 5,000,000 characters
patterns = ["1", "12", "123", "1234", "12345"]

# Measure the time for Boyer-Moore
start_time = time.time()
bm_matches = boyer_moore_search(text, patterns)
bm_time = time.time() - start_time
print(f"\nBoyer-Moore time: {bm_time:.6f} seconds")

# Measure the time for Aho-Corasick
def aho_corasick_search(text, patterns):
    # Initialize the Aho-Corasick Automaton
    A = AhoCorasick(patterns)
    # Search in the text
    return A.search(text)

start_time = time.time()
ac_matches = aho_corasick_search(text, patterns)
ac_time = time.time() - start_time
print(f"Aho-Corasick time: {ac_time:.6f} seconds")

# Measure the time for Wu-Manber
wu_manber = WuManber(patterns)
start_time = time.time()
wu_manber_matches = wu_manber.search(text)
wu_manber_time = time.time() - start_time
print(f"Wu-Manber time: {wu_manber_time:.6f} seconds")\
    
# Measure the time for Trie with Bitwise Indexing
trie = TrieBitwiseIndexing(patterns)
start_time = time.time()
trie_matches = trie.search(text)
trie_time = time.time() - start_time
print(f"Trie with Bitwise Indexing time: {trie_time:.6f} seconds")
