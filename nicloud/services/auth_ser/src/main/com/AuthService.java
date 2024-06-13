package main.com;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@Service
public class AuthService {

    @Value("${jwt.secret}")
    private String secret;

    @Value("${jwt.expiration}")
    private long expirationTime;

    public String login(User user) {
        // Validate user credentials (this is a simple example, use a database in real scenarios)
        if ("user".equals(user.getUsername()) && "password".equals(user.getPassword())) {
            Map<String, Object> claims = new HashMap<>();
            return generateToken(claims, user.getUsername());
        } else {
            throw new RuntimeException("Invalid credentials");
        }
    }

    public String logout(String token) {
        // In a stateless JWT setup, logout can be a client-side operation (e.g., remove token from client storage)
        return "logged out";
    }

    public boolean validateToken(String token) {
        try {
            Claims claims = Jwts.parser()
                    .setSigningKey(secret)
                    .parseClaimsJws(token)
                    .getBody();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private String generateToken(Map<String, Object> claims, Object object) {
        return Jwts.builder()
                .setClaims(claims)
                .setSubject(object)
                .setIssuedAt(new Date(System.currentTimeMillis()))
                .setExpiration(new Date(System.currentTimeMillis() + expirationTime))
                .signWith(SignatureAlgorithm.HS512, secret)
                .compact();
    }
}
