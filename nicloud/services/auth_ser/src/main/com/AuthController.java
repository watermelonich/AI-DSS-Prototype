package main.com;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/auth")
public class AuthController {

    @Autowired
    private ServUser servUser;

    @PostMapping("/login")
    public String login(@RequestBody User user) {
        return servUser.login(user);
    }

    @PostMapping("/logout")
    public String logout(@RequestHeader("Authorization") String token) {
        return servUser.logout(token);
    }

    @GetMapping("/validate")
    public boolean validateToken(@RequestHeader("Authorization") String token) {
        return servUser.validateToken(token);
    }
}
