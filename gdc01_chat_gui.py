"""
GDC-01 Chat GUI - A graphical interface for interacting with the GDC-01 language model.

This creates a modern chat interface using tkinter for conversing with GDC-01,
a Generative Dense Chain language model.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
import numpy as np
from typing import Optional, List

from generative_dense_chain import GenerativeDenseChain
from gdc01_chat import (
    TextEncoder,
    load_conversations,
    build_gdc_model,
    greedy_generate_with_state_update,
    condition_on_text,
    USER_PREFIX,
    GDC01_PREFIX,
    CONVERSATION_DELIMITER
)


class GDC01ChatGUI:
    """
    A graphical chat interface for GDC-01.
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GDC-01 Chat")
        self.root.geometry("700x600")
        self.root.minsize(500, 400)
        
        # Set up colors and styling
        self.bg_color = "#1a1a2e"
        self.chat_bg = "#16213e"
        self.user_color = "#4a9eff"
        self.gdc_color = "#00d9a0"
        self.text_color = "#e8e8e8"
        self.input_bg = "#0f3460"
        self.button_color = "#e94560"
        
        self.root.configure(bg=self.bg_color)
        
        # Model components (loaded asynchronously)
        self.gdc: Optional[GenerativeDenseChain] = None
        self.encoder: Optional[TextEncoder] = None
        self.state_dist: Optional[np.ndarray] = None
        self.conversations: List[str] = []  # Store all training conversations
        self.conversation_history = ""
        self.is_generating = False
        self.training_data_file = "gdc01_training_data.txt"
        
        # Build the UI
        self._create_widgets()
        
        # Load model in background
        self._load_model_async()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.bg_color, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(
            header_frame,
            text="GDC-01",
            font=("Segoe UI", 24, "bold"),
            fg=self.gdc_color,
            bg=self.bg_color
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Generative Dense Chain Language Model",
            font=("Segoe UI", 10),
            fg="#888888",
            bg=self.bg_color
        )
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0), pady=(8, 0))
        
        # Status indicator
        self.status_label = tk.Label(
            header_frame,
            text="Loading model...",
            font=("Segoe UI", 9),
            fg="#ffaa00",
            bg=self.bg_color
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Chat display area
        chat_frame = tk.Frame(main_frame, bg=self.chat_bg, bd=0)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg=self.chat_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            selectbackground="#3a5a8a",
            relief=tk.FLAT,
            padx=15,
            pady=15,
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground=self.user_color, font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("gdc", foreground=self.gdc_color, font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("message", foreground=self.text_color)
        self.chat_display.tag_configure("system", foreground="#888888", font=("Consolas", 10, "italic"))
        
        # Input area
        input_frame = tk.Frame(main_frame, bg=self.bg_color)
        input_frame.pack(fill=tk.X)
        
        # Text input with custom styling (multi-line support)
        input_container = tk.Frame(input_frame, bg=self.input_bg)
        input_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.input_field = tk.Text(
            input_container,
            font=("Segoe UI", 12),
            bg=self.input_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            relief=tk.FLAT,
            bd=0,
            height=2,  # Default height in lines
            wrap=tk.WORD,
            padx=8,
            pady=8
        )
        self.input_field.pack(fill=tk.X, expand=True)
        
        # Key binding: Handle Enter key (check for Shift modifier in handler)
        self.input_field.bind("<Return>", self._on_enter_key)
        
        # Send button
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            font=("Segoe UI", 11, "bold"),
            bg=self.button_color,
            fg="white",
            activebackground="#ff6b80",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=8,
            cursor="hand2",
            command=self._on_send
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Clear button
        self.clear_button = tk.Button(
            input_frame,
            text="Clear",
            font=("Segoe UI", 10),
            bg="#333355",
            fg="#aaaaaa",
            activebackground="#444466",
            activeforeground="white",
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self._clear_chat
        )
        self.clear_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Add to Training Data button
        self.add_training_button = tk.Button(
            input_frame,
            text="Add to Training Data",
            font=("Segoe UI", 10),
            bg="#2d6a4f",
            fg="#ffffff",
            activebackground="#40916c",
            activeforeground="white",
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2",
            command=self._add_to_training_data
        )
        self.add_training_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Info footer
        footer_frame = tk.Frame(main_frame, bg=self.bg_color)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        info_text = "Built by Roberto Dailey | No training required - the model IS the data"
        info_label = tk.Label(
            footer_frame,
            text=info_text,
            font=("Segoe UI", 8),
            fg="#555555",
            bg=self.bg_color
        )
        info_label.pack()
    
    def _append_to_chat(self, text: str, tag: str = "message"):
        """Append text to the chat display with the specified tag."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text, tag)
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)
    
    def _clear_chat(self):
        """Clear the chat, input field, and reset conversation state."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        
        # Clear input field
        self.input_field.delete("1.0", tk.END)
        
        # Reset conversation state
        self.conversation_history = ""
        if self.gdc is not None:
            self.state_dist = self.gdc._get_initial_distribution()
        
        # Show welcome message
        self._show_welcome()
    
    def _show_welcome(self):
        """Show welcome message in chat."""
        welcome = (
            "Welcome to GDC-01!\n"
            "I am a language model built on generative dense chains, "
            "a new type of interpretable sequence model.\n\n"
            "Type a message below to start chatting.\n\n"
        )
        self._append_to_chat(welcome, "system")
    
    def _load_model_async(self):
        """Load the GDC model in a background thread."""
        def load():
            try:
                # Load conversations
                conversations = load_conversations(self.training_data_file)
                
                if not conversations:
                    self.root.after(0, lambda: self._set_status("Error: No training data found", "#ff4444"))
                    return
                
                # Store conversations for later updates
                self.conversations = conversations.copy()
                
                # Build encoder
                self.encoder = TextEncoder()
                self.encoder.fit(conversations)
                
                # Build model
                self.gdc = build_gdc_model(
                    conversations=conversations,
                    encoder=self.encoder,
                    alpha=0.9,
                    theta=0.05,
                    transition_type='self_loop',
                    initial_dist='sequence_starts'
                )
                
                # Initialize state distribution
                self.state_dist = self.gdc._get_initial_distribution()
                
                # Update UI on main thread
                self.root.after(0, self._on_model_loaded)
                
            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {str(e)}", "#ff4444"))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_model_loaded(self):
        """Called when model is loaded successfully."""
        self._set_status(f"Ready ({self.gdc.n_states:,} states)", self.gdc_color)
        self._show_welcome()
        self.input_field.focus_set()
    
    def _set_status(self, text: str, color: str):
        """Update the status label."""
        self.status_label.configure(text=text, fg=color)
    
    def _on_enter_key(self, event):
        """Handle Enter key - send message, or insert newline if Shift is held."""
        # Check if Shift key is held (state & 1 for Shift on most systems)
        shift_held = event.state & 0x0001
        
        if shift_held:
            # Shift+Enter: insert newline
            self.input_field.insert(tk.INSERT, "\n")
            return "break"
        else:
            # Enter alone: send message
            self._on_send()
            return "break"  # Prevent default newline insertion
    
    def _on_send(self, event=None):
        """Handle send button click or Enter key."""
        if self.gdc is None or self.is_generating:
            return
        
        user_text = self.input_field.get("1.0", tk.END).strip()
        if not user_text:
            return
        
        # Clear input field
        self.input_field.delete("1.0", tk.END)
        
        # Display user message
        self._append_to_chat("User: ", "user")
        self._append_to_chat(f"{user_text}\n\n", "message")
        
        # Update conversation history
        user_turn = f"User: {user_text}\n"
        gdc01_start = "GDC-01: "
        self.conversation_history += user_turn + gdc01_start
        
        # Display GDC-01 prefix
        self._append_to_chat("GDC-01: ", "gdc")
        
        # Generate response in background
        self._generate_response_async()
    
    def _generate_response_async(self):
        """Generate response in a background thread."""
        self.is_generating = True
        self._set_status("Generating...", "#ffaa00")
        self.send_button.configure(state=tk.DISABLED)
        
        def generate():
            try:
                # Condition model on full context
                self.state_dist = condition_on_text(
                    self.gdc, 
                    self.encoder, 
                    self.conversation_history
                )
                
                # Generate response character by character
                response, self.state_dist, _ = greedy_generate_with_state_update(
                    gdc=self.gdc,
                    encoder=self.encoder,
                    state_dist=self.state_dist,
                    max_chars=1000,
                    stop_on_user=True
                )
                
                # Update conversation history
                self.conversation_history += response
                if not self.conversation_history.endswith('\n'):
                    self.conversation_history += '\n'
                
                # Display on main thread
                self.root.after(0, lambda: self._display_response(response))
                
            except Exception as e:
                self.root.after(0, lambda: self._display_error(str(e)))
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    def _display_response(self, response: str):
        """Display the generated response."""
        self._append_to_chat(f"{response}\n\n", "message")
        self._set_status(f"Ready ({self.gdc.n_states:,} states)", self.gdc_color)
        self.is_generating = False
        self.send_button.configure(state=tk.NORMAL)
        self.input_field.focus_set()
    
    def _display_error(self, error: str):
        """Display an error message."""
        self._append_to_chat(f"[Error: {error}]\n\n", "system")
        self._set_status("Error", "#ff4444")
        self.is_generating = False
        self.send_button.configure(state=tk.NORMAL)
    
    def _add_to_training_data(self):
        """Add the text from the input field to training data and rebuild the model."""
        if self.gdc is None or self.is_generating:
            return
        
        # Get text from the input field
        conversation = self.input_field.get("1.0", tk.END).strip()
        if not conversation:
            messagebox.showinfo("No Text", "Please enter conversation text in the input field to add to training data.")
            return
        
        # Check if conversation has at least one exchange
        if "User:" not in conversation or "GDC-01:" not in conversation:
            messagebox.showinfo(
                "Invalid Format", 
                "The text should contain at least one 'User:' message and one 'GDC-01:' response.\n\n"
                "Example format:\n"
                "User: Hello!\n"
                "GDC-01: Hi there! I'm GDC-01."
            )
            return
        
        # Confirm with user
        # Show preview (truncate if too long)
        preview = conversation[:200] + "..." if len(conversation) > 200 else conversation
        result = messagebox.askyesno(
            "Add to Training Data",
            f"Add this to training data?\n\n"
            f"Preview:\n{preview}\n\n"
            f"This will:\n"
            f"1. Save to {self.training_data_file}\n"
            f"2. Rebuild the model with the new data\n"
            f"3. Clear the chat and input field\n\n"
            f"Continue?"
        )
        
        if not result:
            return
        
        # Store the conversation text before clearing
        conversation_to_add = conversation
        
        # Clear input field immediately
        self.input_field.delete("1.0", tk.END)
        
        # Disable buttons during update
        self._set_status("Saving...", "#ffaa00")
        self.send_button.configure(state=tk.DISABLED)
        self.add_training_button.configure(state=tk.DISABLED)
        self.clear_button.configure(state=tk.DISABLED)
        
        def update():
            try:
                # Append conversation to the file
                filepath = os.path.join(os.path.dirname(__file__), self.training_data_file)
                
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(CONVERSATION_DELIMITER)
                    f.write(conversation_to_add)
                
                # Add to our conversations list
                self.conversations.append(conversation_to_add)
                
                # Rebuild encoder with all conversations (in case new characters were introduced)
                self.encoder = TextEncoder()
                self.encoder.fit(self.conversations)
                
                # Rebuild model with updated conversations
                self.gdc = build_gdc_model(
                    conversations=self.conversations,
                    encoder=self.encoder,
                    alpha=0.9,
                    theta=0.05,
                    transition_type='self_loop',
                    initial_dist='sequence_starts'
                )
                
                # Reset state distribution
                self.state_dist = self.gdc._get_initial_distribution()
                
                # Update UI on main thread
                self.root.after(0, self._on_training_data_added)
                
            except Exception as e:
                self.root.after(0, lambda: self._on_training_data_error(str(e)))
        
        thread = threading.Thread(target=update, daemon=True)
        thread.start()
    
    def _on_training_data_added(self):
        """Called when training data has been added successfully."""
        # Clear the chat
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        
        # Reset conversation history
        self.conversation_history = ""
        
        # Show success message
        self._append_to_chat(
            f"Conversation added to training data!\n"
            f"Model rebuilt with {len(self.conversations)} conversations "
            f"({self.gdc.n_states:,} states).\n\n",
            "system"
        )
        self._show_welcome()
        
        # Re-enable buttons
        self._set_status(f"Ready ({self.gdc.n_states:,} states)", self.gdc_color)
        self.send_button.configure(state=tk.NORMAL)
        self.add_training_button.configure(state=tk.NORMAL)
        self.clear_button.configure(state=tk.NORMAL)
        self.input_field.focus_set()
    
    def _on_training_data_error(self, error: str):
        """Called when there's an error adding training data."""
        self._append_to_chat(f"[Error saving training data: {error}]\n\n", "system")
        self._set_status("Error", "#ff4444")
        self.send_button.configure(state=tk.NORMAL)
        self.add_training_button.configure(state=tk.NORMAL)
        self.clear_button.configure(state=tk.NORMAL)


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    
    # Set app icon (if available)
    try:
        # Use a default icon if none exists
        pass
    except:
        pass
    
    app = GDC01ChatGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
