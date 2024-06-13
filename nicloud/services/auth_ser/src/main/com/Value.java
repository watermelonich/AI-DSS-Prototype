package main.com;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class Value {

    @Value("${jwt.secret}")
    private String secret;

    @Value("${jwt.expiration}")
    private long expirationTime;

    public String getSecret() {
        return secret;
    }

    public long getExpirationTime() {
        return expirationTime;
    }
}
