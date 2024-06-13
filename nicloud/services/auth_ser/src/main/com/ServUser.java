package main.com;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ServUser {

    @Autowired
    private AuthService authService;

    @Autowired
    private UserRepository userRepository;

    public String login(User user) {
        if (userRepository.validateCredentials(user.getUsername(), user.getPassword())) {
            return authService.login(user);
        } else {
            throw new RuntimeException("Invalid credentials");
        }
    }

    public String logout(String token) {
        return authService.logout(token);
    }

    public boolean validateToken(String token) {
        return authService.validateToken(token);
    }
}
