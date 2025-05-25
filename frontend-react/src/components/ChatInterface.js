// src/components/ChatInterface.js
import React, { useState, useEffect, useRef } from 'react';
import { askQuestion } from '../apiService';
// import './ChatInterface.css'; 
import {
  Box,
  VStack,
  HStack,
  Input,
  // Button, // Removed as IconButton is used for send
  Text,
  Heading,
  List,
  ListItem,
  useToast, 
  Spinner, 
  Flex, 
  Avatar, 
  Tag, 
  Tooltip, 
  IconButton, 
} from '@chakra-ui/react';
import { ChatIcon, InfoOutlineIcon } from '@chakra-ui/icons';

function ChatInterface() {
  const [messages, setMessages] = useState([
    { sender: 'assistant', text: "Hello! Ask me anything about the loaded documents." }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const toast = useToast(); 

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { sender: 'user', text: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    let errorMessageText = 'Failed to get a response from the assistant.'; // Default error
    try {
      const response = await askQuestion(currentInput);
      const assistantMessage = {
        sender: 'assistant',
        text: response.answer,
        sources: response.source_documents || [],
      };
      setMessages((prevMessages) => [...prevMessages, assistantMessage]);
    } catch (err) {
      console.error("Error fetching response:", err);
      if (err instanceof Error) {
        errorMessageText = err.message;
      } else if (typeof err === 'string') {
        errorMessageText = err;
      } else if (err && typeof err === 'object' && err.detail) {
        try {
          if (Array.isArray(err.detail) && err.detail.length > 0 && err.detail[0].msg) {
            errorMessageText = err.detail[0].msg;
          } else if (typeof err.detail === 'string'){
            errorMessageText = err.detail;
          }
        } catch (parseError) {
          console.error("Could not parse error detail:", parseError);
        }
      }
      
      toast({ 
        title: "Error",
        description: errorMessageText, 
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      
      const errorMessageForChat = { 
        sender: 'assistant', 
        text: `Error: ${errorMessageText}`, 
        isError: true 
      };
      setMessages((prevMessages) => [...prevMessages, errorMessageForChat]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Flex 
      direction="column" 
      h="100vh" 
      maxW="800px" 
      mx="auto" 
      borderWidth={{ base: "0", md: "1px" }} 
      borderColor="gray.200"
      borderRadius={{ base: "0", md: "lg" }}
      boxShadow={{ base: "none", md: "lg" }}
      overflow="hidden"
    >
      <Box bg="blue.500" color="white" p={4} textAlign="center">
        <Heading size="lg">RAG Assistant Chat</Heading>
      </Box>

      <VStack 
        spacing={4} 
        p={5} 
        flexGrow={1} 
        overflowY="auto" 
        alignItems="stretch"
        bg="gray.50"
      >
        {messages.map((msg, index) => (
          <Flex 
            key={index} 
            w="100%" 
            justify={msg.sender === 'user' ? 'flex-end' : 'flex-start'}
          >
            <HStack 
              spacing={3} 
              alignItems="flex-start" 
              flexDir={msg.sender === 'user' ? 'row-reverse' : 'row'}
            >
              <Avatar 
                size="sm" 
                name={msg.sender === 'user' ? 'User' : 'Assistant'} 
                src={msg.sender === 'user' ? undefined : undefined} 
                bg={msg.sender === 'user' ? 'blue.300' : 'gray.300'}
              />
              <Box 
                bg={msg.sender === 'user' ? 'blue.500' : (msg.isError ? 'red.100' : 'white')}
                color={msg.sender === 'user' ? 'white' : (msg.isError ? 'red.700' : 'gray.800')}
                px={4} 
                py={2} 
                borderRadius="lg"
                boxShadow="sm"
                maxW="70%"
              >
                <Text 
                  whiteSpace="pre-wrap"
                  color={msg.sender === 'user' ? 'white' : (msg.isError ? 'red.700' : 'gray.800')}
                > 
                  {msg.text}
                </Text>
                {msg.sender === 'assistant' && !msg.isError && msg.sources && msg.sources.length > 0 && (
                  <Box mt={2} fontSize="sm">
                    <Text fontWeight="bold" mb={1} color="gray.600">Sources:</Text> 
                    <List spacing={1}>
                      {msg.sources.map((source, idx) => (
                        <ListItem key={idx} d="flex" alignItems="center">
                          <InfoOutlineIcon mr={2} color="gray.500" /> 
                          <Tooltip 
                            label={source.content?.substring(0, 300) + (source.content?.length > 300 ? '...' : '')}
                            aria-label={`Source: ${source.metadata?.source || 'Unknown'}`}
                            bg="gray.700"
                            color="white"
                            p={2}
                            borderRadius="md"
                          >
                            <Tag size="sm" variant="subtle" colorScheme="gray">
                              {source.metadata?.source ? source.metadata.source.split(/[\\/]/).pop() : 'Unknown Source'}
                              {source.metadata?.page !== undefined && source.metadata.page !== null && ` (Page: ${source.metadata.page})`}
                            </Tag>
                          </Tooltip>
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </Box>
            </HStack>
          </Flex>
        ))}
        <Box ref={messagesEndRef} />
      </VStack>

      <Box p={4} borderTopWidth="1px" borderColor="gray.200" bg="white">
        <form onSubmit={handleSubmit}>
          <HStack spacing={3}>
            <Input
              variant="filled"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={isLoading}
              size="lg"
              borderRadius="full"
              color="gray.800" 
              _placeholder={{ color: 'gray.500' }} 
            />
            <IconButton 
              colorScheme="blue"
              aria-label="Send message"
              icon={isLoading ? <Spinner size="sm" /> : <ChatIcon />}
              type="submit"
              isDisabled={isLoading || !input.trim()}
              isRound
              size="lg"
            />
          </HStack>
        </form>
      </Box>
    </Flex>
  );
}

export default ChatInterface;
